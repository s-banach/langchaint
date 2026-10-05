"""Verify kind values across langchaint tagged unions.

Discovery reads type aliases from each langchaint module.
Each tagged union has distinct kind values.
"""

import inspect
from types import UnionType
from typing import TypeAliasType, Union, get_args, get_origin

from tests.helpers import package_modules


def _flatten_union(annotation: object) -> list[type]:
    """Return concrete classes from an annotation."""
    aliased = getattr(annotation, "__value__", None)
    if aliased is not None:
        return _flatten_union(aliased)
    if getattr(annotation, "__metadata__", None) is not None:
        return _flatten_union(getattr(annotation, "__origin__", None))
    # Python 3.13 uses typing.Union or types.UnionType based on the operands.
    if get_origin(annotation) in (Union, UnionType):
        return [
            variant for argument in get_args(annotation) for variant in _flatten_union(argument)
        ]
    origin = get_origin(annotation)
    if isinstance(origin, type):
        return [origin]
    return [annotation] if isinstance(annotation, type) else []


def _tagged_unions() -> dict[str, tuple[type, ...]]:
    """Map each tagged type alias to its variants."""
    found: dict[str, tuple[type, ...]] = {}
    for module in package_modules():
        for attribute_name, value in vars(module).items():
            if getattr(value, "__value__", None) is None:
                continue
            variants = _flatten_union(value)
            if variants and all(
                "kind" in inspect.get_annotations(variant) for variant in variants
            ):
                found[attribute_name] = tuple(variants)
    return found


def _tags_of(variant: type) -> tuple[str, ...]:
    """Return the strings in a variant's kind Literal, which a type alias may name."""
    annotation = inspect.get_annotations(variant)["kind"]
    if isinstance(annotation, TypeAliasType):
        annotation = annotation.__value__
    tags = get_args(annotation)
    assert tags, f"{variant.__name__}.kind holds no Literal value"
    assert all(isinstance(tag, str) for tag in tags), (
        f"{variant.__name__}.kind is not a string Literal"
    )
    return tags


_TAGGED_UNIONS = _tagged_unions()


def test_no_union_gives_two_variants_the_same_tag() -> None:
    """Each kind value identifies one variant of a tagged union."""
    assert _TAGGED_UNIONS
    collisions = {}
    for union_name, variants in _TAGGED_UNIONS.items():
        tags = [tag for variant in variants for tag in _tags_of(variant)]
        duplicated = sorted({tag for tag in tags if tags.count(tag) > 1})
        if duplicated:
            collisions[union_name] = duplicated
    assert not collisions
