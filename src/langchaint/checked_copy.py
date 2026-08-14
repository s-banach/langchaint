"""A pydantic base that rejects unknown keys during construction and `model_copy`.

Subclasses must forbid extra keys and use pydantic's generated constructor.
`model_construct` retains pydantic's unvalidated behavior.
"""

from collections.abc import Mapping
from typing import Self, override

from pydantic import BaseModel


def _bad_update_key_message(model_class: type[BaseModel], key: str) -> str:
    if isinstance(getattr(model_class, key, None), property):
        return (
            f"model_copy update key {key!r} is a derived property of {model_class.__name__}, "
            f"computed from its fields; construct a new {model_class.__name__} "
            "from changed fields instead of updating the view"
        )
    return (
        f"model_copy update key {key!r} is not a field of {model_class.__name__}; "
        f"fields: {sorted(model_class.model_fields)}"
    )


class CheckedCopyModel(BaseModel):
    """Reject unknown keys during construction and `model_copy`."""

    @classmethod
    @override
    def __pydantic_init_subclass__(cls, **kwargs: object) -> None:
        """Require `extra="forbid"` and pydantic's generated constructor.

        Raises:
            TypeError: The subclass permits extra keys or defines `__init__`.
        """
        super().__pydantic_init_subclass__(**kwargs)
        extra = cls.model_config.get("extra")
        if extra == "allow":
            raise TypeError(
                f"{cls.__name__} sets extra='allow', under which a key that is not a field is kept "
                "in __pydantic_extra__ as meaningful data, so CheckedCopyModel.model_copy would "
                "reject legitimate updates; such a model must not inherit CheckedCopyModel"
            )
        if extra != "forbid":
            states = (
                f"it sets extra={extra!r}"
                if extra is not None
                else "it sets no extra, leaving pydantic's 'ignore' default"
            )
            raise TypeError(
                f"{cls.__name__} must set model_config = ConfigDict(extra='forbid'); {states}, under "
                "which a key that is not a field is dropped silently on construction instead of raising"
            )
        if cls.__pydantic_custom_init__:
            raise TypeError(
                f"{cls.__name__} defines __init__, which sets pydantic's custom_init: pydantic "
                "binds the raw input to that signature, so a surplus or missing key is rejected by "
                "argument binding as an unlocated TypeError before extra='forbid' is consulted. "
                "Let pydantic generate the constructor; its arguments are keyword-only"
            )

    @override
    def model_copy(
        self, *, update: Mapping[str, object] | None = None, deep: bool = False
    ) -> Self:
        """Copy after checking that every update key names a field.

        Field values retain pydantic's unvalidated `model_copy` behavior.

        Raises:
            TypeError: An update key does not name a field.
        """
        if update:
            for key in update:
                if key not in type(self).model_fields:
                    raise TypeError(_bad_update_key_message(type(self), key))
        return super().model_copy(update=update, deep=deep)
