"""Exercise the annotations available to callers through the public APIs."""

import pkgutil
from collections.abc import Callable, Iterator
from functools import cached_property
from importlib import import_module
from itertools import chain
from types import FunctionType, ModuleType
from typing import get_overloads, get_type_hints

import langchaint
from langchaint import conformance


def _exports(module: ModuleType) -> Iterator[object]:
    if module is conformance:
        yield from (
            value
            for name, value in vars(module).items()
            if not name.startswith("_")
            and isinstance(value, type)
            and value.__module__ == module.__name__
        )
    for name in vars(module).get("__all__", ()):
        yield getattr(module, name)


def _functions(value: object) -> Iterator[Callable[..., object]]:
    if isinstance(value, property):
        yield from (
            function for function in (value.fget, value.fset, value.fdel) if function is not None
        )
    elif isinstance(value, cached_property):
        yield value.func
    elif isinstance(value, (staticmethod, classmethod)):
        yield value.__func__
    elif isinstance(value, FunctionType):
        yield value


def _annotations(value: object) -> Iterator[type | Callable[..., object]]:
    if isinstance(value, type):
        yield value
        for base in value.__mro__:
            if base.__module__.split(".")[0] != "langchaint":
                continue
            for name, attribute in vars(base).items():
                if not name.startswith("_") or name.startswith("__"):
                    yield from _functions(attribute)
    else:
        yield from _functions(value)


def test_public_annotations_resolve() -> None:
    modules = (
        import_module(info.name)
        for info in pkgutil.iter_modules(langchaint.__path__, prefix="langchaint.")
    )
    exports = chain.from_iterable(map(_exports, chain([langchaint], modules)))
    for target in set(chain.from_iterable(map(_annotations, exports))):
        for annotated in (target, *get_overloads(target)):
            try:
                _ = get_type_hints(annotated, include_extras=True)
            except (NameError, TypeError) as error:
                raise AssertionError(
                    f"{annotated.__module__}.{annotated.__qualname__}: {error}"
                ) from error
