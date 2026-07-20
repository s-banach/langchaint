"""CheckedCopyModel, the base of langchaint's pydantic models: model_copy rejects non-field update keys.

pydantic's model_copy applies update without validation:
a key that is not a field lands in the instance __dict__,
where any class-level property of that name shadows it and model_dump ignores it,
so the caller's intended change is dropped silently.
Under langchaint's model configuration (frozen, no extra fields, no private attributes)
a non-field update key therefore can never do anything,
and the override turns that defect into an immediate TypeError,
matching dataclasses.replace, which raises TypeError for an unknown field name on the same operation shape.
A subclass setting extra="allow" is rejected at class definition:
that configuration is the one under which unknown update keys become meaningful
(pydantic stores them in __pydantic_extra__), so the check would reject legitimate updates on such a model.
"""

from collections.abc import Mapping
from typing import Self, override

from pydantic import BaseModel


def _bad_update_key_message(model_class: type[BaseModel], key: str) -> str:
    """Explain one rejected update key: a derived property gets its specific fix, anything else the field list."""
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
    """Base class whose model_copy raises on update keys that are not model fields."""

    @classmethod
    @override
    def __pydantic_init_subclass__(cls, **kwargs: object) -> None:
        """Reject a subclass whose configuration would make the model_copy check misfire.

        The **kwargs pass-through is the hook's signature: pydantic forwards class-definition keyword arguments.

        Raises:
            TypeError: the subclass sets extra="allow", whose legitimate extra-key updates
                model_copy would wrongly reject as non-field keys.
        """
        super().__pydantic_init_subclass__(**kwargs)
        if cls.model_config.get("extra") == "allow":
            raise TypeError(
                f"{cls.__name__} sets extra='allow', whose legitimate extra-key updates "
                "CheckedCopyModel.model_copy would wrongly reject; do not inherit the guard"
            )

    @override
    def model_copy(self, *, update: Mapping[str, object] | None = None, deep: bool = False) -> Self:
        """Copy with the update keys checked against the model's fields before pydantic's unvalidated copy.

        Field values still bypass validation, as on pydantic's model_copy:
        a wrong value on a legitimate field key is stored and visible at first use,
        unlike a dropped key, and catching it would revalidate the nested tree on every copy.

        Raises:
            TypeError: an update key is not a field of this model,
                so pydantic's unvalidated copy would drop it silently instead of applying it.
        """
        if update:
            for key in update:
                if key not in type(self).model_fields:
                    raise TypeError(_bad_update_key_message(type(self), key))
        return super().model_copy(update=update, deep=deep)
