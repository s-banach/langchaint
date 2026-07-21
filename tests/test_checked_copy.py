"""The two class-definition checks behind CheckedCopyModel, and the construction rule they enforce.

CheckedCopyModel covers a model only if that model inherits it,
so a model declared on plain BaseModel would take neither the model_copy guard nor extra="forbid".
The first test walks every module under langchaint and fails on any package-defined pydantic model
outside the base. There is deliberately no allowlist: a model that must not inherit the base
(say one that sets extra="allow", whose legitimate extra-key updates model_copy would reject)
fails here and forces that design discussion instead of slipping in undocumented.
The rest cover __pydantic_init_subclass__, which requires each subclass's effective config to be
extra="forbid" (pydantic merges model_config from bases, so a subclass of a model that sets it
inherits it), and the construction-time rejection that setting buys.
"""

import importlib
import inspect
import pkgutil
from collections.abc import Iterator
from types import ModuleType

import pytest
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

import langchaint
from langchaint.checked_copy import CheckedCopyModel
from langchaint.messages import Message
from langchaint.usage import Usage


def _package_modules() -> Iterator[ModuleType]:
    """Import every module under langchaint, backend subpackages included.

    Importing the backend subpackages and tracing requires both SDKs and opentelemetry-api,
    which the dev environment installs for the adapter and tracing tests.

    Yields:
        Each imported module, the package itself first.
    """
    yield langchaint
    for module_info in pkgutil.walk_packages(langchaint.__path__, prefix="langchaint."):
        yield importlib.import_module(module_info.name)


def test_every_package_pydantic_model_inherits_checked_copy_model() -> None:
    """Every pydantic model defined in the package inherits CheckedCopyModel.

    The __module__ comparison keeps only classes defined in the walked module,
    so SDK pydantic models imported by the adapters and the package's own re-exports are skipped.
    """
    offenders = [
        f"{cls.__module__}.{cls.__name__}"
        for module in _package_modules()
        for _name, cls in inspect.getmembers(module, inspect.isclass)
        if cls.__module__ == module.__name__
        and issubclass(cls, BaseModel)
        and not issubclass(cls, CheckedCopyModel)
    ]
    assert offenders == []


def test_a_subclass_defining_its_own_init_is_rejected() -> None:
    """A subclass must leave __init__ to pydantic; the checked_copy module docstring says why.

    The hook checks this rather than a walk over the package's own models, because it fires at
    class definition and so covers a subclass an application defines outside langchaint.
    The second shape is a positional-only receiver plus **extra, which does route a surplus key past
    argument binding, and is rejected anyway, because a missing field still binds to the signature
    and raises TypeError there.
    """
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
    """Under extra="allow" unknown update keys are meaningful, so the guard would reject legitimate copies.

    CheckedCopyModel therefore raises from __pydantic_init_subclass__ the moment such a subclass is defined.
    """
    with pytest.raises(TypeError, match="extra='allow'"):

        class ExtraAllowModel(CheckedCopyModel):
            model_config = ConfigDict(extra="allow")


def test_a_subclass_leaving_extra_unset_is_rejected_at_class_definition() -> None:
    """Omitting extra leaves pydantic's "ignore" default, under which construction drops unknown keys.

    This is the other half of the rule, and the common way to break it: a new model that simply
    does not think about extra reads as fine until a misspelled key goes missing at runtime.
    """
    with pytest.raises(TypeError, match="no extra"):

        class DefaultExtraModel(CheckedCopyModel):
            value: int


def test_a_subclass_setting_extra_ignore_is_rejected_at_class_definition() -> None:
    """Stating the "ignore" default explicitly is rejected the same way as omitting it.

    The two reach the same pydantic behavior by different routes, so the message differs (it names
    the value rather than the default) while the rejection does not.
    """
    with pytest.raises(TypeError, match="extra='ignore'"):

        class ExtraIgnoreModel(CheckedCopyModel):
            model_config = ConfigDict(extra="ignore")


def test_a_subclass_inheriting_forbid_from_its_base_passes_without_restating_it() -> None:
    """The hook reads the merged config, so a model built on another model needs no second line.

    Reading only the class's own model_config would pass every other test here, since no langchaint
    model subclasses another today, and would reject the first one that does.
    """

    class Base(CheckedCopyModel):
        model_config = ConfigDict(extra="forbid")
        value: int

    class Child(Base):
        other: int

    with pytest.raises(ValidationError, match="junk"):
        Child(value=1, other=2, junk=3)  # pyrefly: ignore[unexpected-keyword]


def test_construction_rejects_a_key_that_is_not_a_field() -> None:
    """The point of requiring extra="forbid": a misspelled field name raises instead of vanishing.

    Usage stands in for every model here, which pydantic constructs the same way; the
    class-definition hook above is what keeps the rest of them from regressing.
    """
    with pytest.raises(ValidationError, match="inpit_tokens_cache_read"):
        Usage(
            input_tokens_cache_read=0,
            input_tokens_cache_write=0,
            input_tokens_cache_none=1,
            output_tokens=1,
            output_tokens_reasoning=0,
            cost_in_usd=0.0,
            inpit_tokens_cache_read=1,  # pyrefly: ignore[unexpected-keyword]
        )


@pytest.mark.parametrize(
    ("payload", "key", "error_type"),
    [
        ({"content": "x", "role": "user", "junk": 1}, "junk", "extra_forbidden"),
        ({"turn": [{"text": "a"}], "role": "assistant", "junk": 1}, "junk", "extra_forbidden"),
        ({"role": "user"}, "content", "missing"),
        ({"role": "assistant"}, "turn", "missing"),
    ],
)
def test_reloading_a_malformed_conversation_locates_the_key_as_a_validation_error(
    payload: dict[str, object], key: str, error_type: str
) -> None:
    """A surplus key and a missing field both raise ValidationError naming where they are.

    This is the reload path for a persisted conversation, so one exception type across the tree
    is what lets an application catch every malformed message with ValidationError and read its
    location. A message model defining its own __init__ breaks both halves: pydantic binds the
    payload to that signature first, and the binding failure is a TypeError naming no location.
    """
    message_type_adapter = TypeAdapter[Message](Message)
    with pytest.raises(ValidationError, match=key) as caught:
        message_type_adapter.validate_python(payload)
    assert [(error["loc"][-1], error["type"]) for error in caught.value.errors()] == [
        (key, error_type)
    ]
