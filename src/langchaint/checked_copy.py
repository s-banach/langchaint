"""CheckedCopyModel, the base of langchaint's pydantic models: a key that is not a field is an error.

Construction and validation are covered by extra="forbid", which the hook below requires of every
subclass: pydantic's "ignore" default would drop a misspelled field name silently, leaving an object
without the value its caller supplied. The cost is that a conversation written by a newer langchaint
raises when an older one loads it, instead of the added field being discarded.
The hook also rejects a subclass that defines __init__, so every model here keeps pydantic's
generated keyword-only constructor and one error shape. A positional constructor, the custom
__init__ that would let a message read UserMessage("Hello"), is rejected: pydantic binds the raw
input to such a signature before extra="forbid" is consulted, so both a surplus key and a missing
field raise TypeError naming no location. Giving that __init__ a **extra parameter routes a surplus
key past binding but does nothing for a missing field, and absorbs a misspelled optional argument
that the generated constructor would have failed at check time.
model_copy needs its own override, because it applies update without validation and so never
consults extra: the key would land in the instance __dict__, where a same-named property shadows it
and model_dump ignores it. It raises TypeError, as dataclasses.replace does for an unknown field.
model_construct is left as pydantic ships it, dropping the key silently, since it exists to skip
validation on data the caller vouches for.
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
    """Base class on which a key that is not a field raises, whether it arrives by construction or copy."""

    @classmethod
    @override
    def __pydantic_init_subclass__(cls, **kwargs: object) -> None:
        """Require the subclass to forbid extra keys and to leave __init__ to pydantic.

        Together those two make construction reject a key that is not a field, with a located
        ValidationError.
        The **kwargs pass-through is the hook's signature: pydantic forwards class-definition keyword arguments.

        The config is checked before __init__, so each message names a fix that is the whole
        remaining fix.

        Raises:
            TypeError: the subclass does not set extra="forbid". Leaving pydantic's "ignore"
                default drops a misspelled key silently, and the fix is to set "forbid"; "allow"
                keeps unknown keys as meaningful data that model_copy would wrongly reject, and
                the fix is to not inherit this base.
            TypeError: the subclass defines its own __init__, which pydantic binds the raw input to
                before extra="forbid" is consulted, so a bad key raises an unlocated TypeError.
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
