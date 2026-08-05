"""The shape of the kind tag, checked structurally over every tagged union langchaint defines.

The unions are discovered by importing every langchaint module and reading its type aliases, so a
union added anywhere is covered the moment it exists. A tagged union is one whose variants all
annotate kind, so a union with a builtin variant, which cannot hold a tag, is out of scope rather
than a failure.

Two invariants hold whatever the union: two variants never share a tag, and every word of a tag comes
from its own class's name. Together they are what lets a match on kind route each variant to its own
case.
The naming check is a subsequence test because a variant drops the words its siblings all share
(UserMessage tags "user"), and a variant of two unions carries one tag under both, so the words
dropped are not a function of any single union's membership.
"""

import inspect
import re
from collections.abc import Sequence
from types import UnionType
from typing import Union, get_args, get_origin

from tests.helpers import package_modules


def _flatten_union(annotation: object) -> list[type]:
    """List the concrete classes an annotation resolves to, unwrapping aliases and generics."""
    aliased = getattr(annotation, "__value__", None)
    if aliased is not None:
        return _flatten_union(aliased)
    if getattr(annotation, "__metadata__", None) is not None:
        return _flatten_union(getattr(annotation, "__origin__", None))
    # Both origins occur and are distinct objects on 3.13: | builds typing.Union when an operand is
    # a typing object, and types.UnionType otherwise.
    if get_origin(annotation) in (Union, UnionType):
        return [
            variant for argument in get_args(annotation) for variant in _flatten_union(argument)
        ]
    origin = get_origin(annotation)
    if isinstance(origin, type):
        return [origin]
    return [annotation] if isinstance(annotation, type) else []


def _own_annotations(variant: type) -> dict[str, object]:
    """Read the annotations the class declares itself, ignoring any it inherits.

    inspect.get_annotations reads them wherever the interpreter keeps them, and evaluates any it
    finds deferred. A 3.13 class dict holds __annotations__; a 3.14 one holds __annotate_func__ and
    caches under __annotations_cache__, so reading __annotations__ out of the dict finds nothing.
    """
    return inspect.get_annotations(variant)


def _tagged_unions() -> dict[str, tuple[type, ...]]:
    """Map each langchaint type alias whose variants all declare kind to those variants."""
    found: dict[str, tuple[type, ...]] = {}
    for module in package_modules():
        for attribute_name, value in vars(module).items():
            if getattr(value, "__value__", None) is None:
                continue
            variants = _flatten_union(value)
            if variants and all("kind" in _own_annotations(variant) for variant in variants):
                found[attribute_name] = tuple(variants)
    return found


def _tag_of(variant: type) -> str:
    """Read the one string a variant's kind Literal holds."""
    literal_arguments = get_args(_own_annotations(variant)["kind"])
    assert len(literal_arguments) == 1, f"{variant.__name__}.kind holds no single Literal value"
    tag = literal_arguments[0]
    assert isinstance(tag, str), f"{variant.__name__}.kind is not a string Literal"
    return tag


def _words(class_name: str) -> list[str]:
    """Split a CamelCase class name into its lowercased words."""
    return [word.lower() for word in re.findall(r"[A-Z][a-z0-9]*", class_name)]


def _is_word_subsequence(tag_words: Sequence[str], class_words: Sequence[str]) -> bool:
    """Report whether every tag word appears in the class words, in order."""
    remaining = list(class_words)
    for word in tag_words:
        if word not in remaining:
            return False
        remaining = remaining[remaining.index(word) + 1 :]
    return True


_TAGGED_UNIONS = _tagged_unions()


def test_discovery_finds_the_known_unions_with_their_variants_intact() -> None:
    """Discovery reaches both the pydantic unions and the outcome unions, and every variant of each.

    Every check below passes vacuously on a union discovery dropped, and passes just as quietly on
    one it found but flattened to a single variant, so a walk that stops early has to fail here.
    A union of one variant is not a union, which is what makes the lower bound safe to assert.
    """
    assert {"Message", "Part", "TurnElement", "ResponseOutcome", "GenerateResult"} <= set(
        _TAGGED_UNIONS
    )
    undersized = {
        name: [variant.__name__ for variant in variants]
        for name, variants in _TAGGED_UNIONS.items()
        if len(variants) < 2
    }
    assert not undersized


def test_no_union_gives_two_variants_the_same_tag() -> None:
    """Within one union every tag is distinct, so a tag identifies one variant.

    A variant that copied a sibling's tag would take the sibling's case in every match on kind,
    leaving its own case dead, and pyrefly reports neither the duplicate nor the dead case.
    """
    collisions = {}
    for union_name, variants in _TAGGED_UNIONS.items():
        tags = [_tag_of(variant) for variant in variants]
        duplicated = sorted({tag for tag in tags if tags.count(tag) > 1})
        if duplicated:
            collisions[union_name] = duplicated
    assert not collisions


def test_every_tag_is_built_from_its_own_class_name() -> None:
    """A tag's words appear in its class's name, in order."""
    variants = {
        variant for union_variants in _TAGGED_UNIONS.values() for variant in union_variants
    }
    misnamed = sorted(
        (variant.__name__, _tag_of(variant))
        for variant in variants
        if not _is_word_subsequence(_tag_of(variant).split("_"), _words(variant.__name__))
    )
    assert not misnamed
