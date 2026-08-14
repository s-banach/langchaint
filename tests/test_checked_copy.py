"""Test CheckedCopyModel class and construction checks.

Each subclass uses extra="forbid" and pydantic's __init__.
Construction rejects keys that are not fields.
"""

import inspect

import pytest
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from langchaint.checked_copy import CheckedCopyModel
from langchaint.messages import Message
from langchaint.usage import Usage
from tests.helpers import package_modules


def test_every_package_pydantic_model_inherits_checked_copy_model() -> None:
    """Each langchaint pydantic model inherits CheckedCopyModel."""
    offenders = [
        f"{cls.__module__}.{cls.__name__}"
        for module in package_modules()
        for _name, cls in inspect.getmembers(module, inspect.isclass)
        if cls.__module__ == module.__name__
        and issubclass(cls, BaseModel)
        and not issubclass(cls, CheckedCopyModel)
    ]
    assert offenders == []


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

    with pytest.raises(ValidationError, match="junk"):
        _ = Child.model_validate({"value": 1, "other": 2, "junk": 3})


def test_construction_rejects_a_key_that_is_not_a_field() -> None:
    """Construction rejects a misspelled field name."""
    with pytest.raises(ValidationError, match="inpit_tokens_cache_read"):
        _ = Usage.model_validate({
            "input_tokens_cache_read": 0,
            "input_tokens_cache_write": 0,
            "input_tokens_cache_none": 1,
            "output_tokens": 1,
            "output_tokens_reasoning": 0,
            "input_tokens_cache_read_cost_in_usd": 0.0,
            "input_tokens_cache_write_cost_in_usd": 0.0,
            "input_tokens_cache_none_cost_in_usd": 0.0,
            "output_tokens_cost_in_usd": 0.0,
            "provider_executed_tool_cost_in_usd": 0.0,
            "inpit_tokens_cache_read": 1,
        })


@pytest.mark.parametrize(
    ("payload", "key", "error_type"),
    [
        ({"content": "x", "kind": "user", "junk": 1}, "junk", "extra_forbidden"),
        (
            {"turn": [{"text": "a", "kind": "text"}], "kind": "assistant", "junk": 1},
            "junk",
            "extra_forbidden",
        ),
        ({"kind": "user"}, "content", "missing"),
        ({"kind": "assistant"}, "turn", "missing"),
    ],
)
def test_reloading_a_malformed_message_locates_the_key_as_a_validation_error(
    payload: dict[str, object], key: str, error_type: str
) -> None:
    """Message validation errors identify surplus and missing fields."""
    message_type_adapter = TypeAdapter[Message](Message)
    with pytest.raises(ValidationError, match=key) as caught:
        _ = message_type_adapter.validate_python(payload)
    assert [(error["loc"][-1], error["type"]) for error in caught.value.errors()] == [
        (key, error_type)
    ]
