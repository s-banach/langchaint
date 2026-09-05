"""Test CheckedCopyModel subclass configuration checks."""

import pytest
from pydantic import ConfigDict

from langchaint.common.checked_copy import CheckedCopyModel


def test_a_subclass_defining_its_own_init_is_rejected() -> None:
    """CheckedCopyModel rejects subclasses with a custom __init__."""
    with pytest.raises(TypeError, match="custom_init"):

        class CustomInitModel(CheckedCopyModel):
            model_config = ConfigDict(extra="forbid")
            value: int

            def __init__(self, value: int) -> None:
                super().__init__(value=value)

    with pytest.raises(TypeError, match="custom_init"):

        class ForwardingInitModel(CheckedCopyModel):
            model_config = ConfigDict(extra="forbid")
            value: int

            def __init__(self, /, value: int, **extra: object) -> None:
                super().__init__(value=value, **extra)


def test_a_subclass_setting_extra_allow_is_rejected_at_class_definition() -> None:
    """CheckedCopyModel rejects extra="allow"."""
    with pytest.raises(TypeError, match="extra='allow'"):

        class ExtraAllowModel(CheckedCopyModel):
            model_config = ConfigDict(extra="allow")


def test_a_subclass_leaving_extra_unset_is_rejected_at_class_definition() -> None:
    """CheckedCopyModel rejects pydantic's default extra behavior."""
    with pytest.raises(TypeError, match="no extra"):

        class DefaultExtraModel(CheckedCopyModel):
            value: int


def test_a_subclass_setting_extra_ignore_is_rejected_at_class_definition() -> None:
    """CheckedCopyModel rejects extra="ignore"."""
    with pytest.raises(TypeError, match="extra='ignore'"):

        class ExtraIgnoreModel(CheckedCopyModel):
            model_config = ConfigDict(extra="ignore")


def test_a_subclass_inheriting_forbid_from_its_base_passes_without_restating_it() -> None:
    """A subclass may inherit extra="forbid" from its base."""

    class Base(CheckedCopyModel):
        model_config = ConfigDict(extra="forbid")
        value: int

    class Child(Base):
        other: int

    _ = Child(value=1, other=2)
