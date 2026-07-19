"""The package-wide inheritance check behind CheckedCopyModel.

CheckedCopyModel guards model_copy only for models that inherit it,
so a model declared on plain BaseModel would silently drop non-field update keys.
This test walks every module under langchaint and fails on any package-defined pydantic model outside the base.
There is deliberately no allowlist: a model that must not inherit the guard
(say one that sets extra="allow", whose legitimate extra-key updates the guard would reject)
fails here and forces that design discussion instead of slipping in undocumented.
"""

import importlib
import inspect
import pkgutil
from collections.abc import Iterator
from types import ModuleType

import pytest
from pydantic import BaseModel, ConfigDict

import langchaint
from langchaint.checked_copy import CheckedCopyModel


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


def test_a_subclass_setting_extra_allow_is_rejected_at_class_definition() -> None:
    """Under extra="allow" unknown update keys are meaningful, so the guard would reject legitimate copies.

    CheckedCopyModel therefore raises from __pydantic_init_subclass__ the moment such a subclass is defined.
    """
    with pytest.raises(TypeError, match="extra='allow'"):

        class ExtraAllowModel(CheckedCopyModel):
            model_config = ConfigDict(extra="allow")
