"""Test CheckedCopyModel subclass configuration checks."""

from typing import Literal

import pytest
from pydantic import ConfigDict, ValidationError

from langchaint.common.checked_copy import CheckedCopyModel


class _Valued(CheckedCopyModel):
    value: int


def test_a_subclass_defining_its_own_init_is_rejected() -> None:
    with pytest.raises(TypeError, match="pydantic's generated constructor"):

        class CustomInitModel(_Valued):
            def __init__(self, value: int) -> None:
                super().__init__(value=value)


@pytest.mark.parametrize("extra", ["allow", "ignore"])
def test_a_subclass_overriding_extra_is_rejected_at_class_definition(
    extra: Literal["allow", "ignore"],
) -> None:
    with pytest.raises(TypeError, match=f"extra='{extra}'"):

        class OverridingModel(CheckedCopyModel):
            model_config = ConfigDict(extra=extra)


def test_a_subclass_with_its_own_config_stays_frozen_and_rejects_unknown_keys() -> None:

    class Child(_Valued):
        model_config = ConfigDict(ser_json_inf_nan="strings")
        other: int

    _ = hash(Child(value=1, other=2))
    with pytest.raises(ValidationError, match="extra_forbidden"):
        _ = Child.model_validate({"value": 1, "other": 2, "unknown": 3})
