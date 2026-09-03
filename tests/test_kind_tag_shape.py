"""Verify kind values across langchaint tagged unions.

Discovery reads type aliases from each langchaint module.
Each tagged union has distinct kind values.
Each kind value contains an ordered subset of its class name's words.
"""

import inspect
import re
from types import UnionType
from typing import Union, get_args, get_origin

from langchaint import GenerationErrorKind
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


def _tag_of(variant: type) -> str:
    """Return the string in a variant's kind Literal."""
    annotations = inspect.get_annotations(variant)
    literal_arguments = get_args(annotations["kind"])
    assert len(literal_arguments) == 1, f"{variant.__name__}.kind holds no single Literal value"
    tag = literal_arguments[0]
    assert isinstance(tag, str), f"{variant.__name__}.kind is not a string Literal"
    return tag


def _words(class_name: str) -> list[str]:
    """Split a CamelCase class name into its lowercased words."""
    return [word.lower() for word in re.findall(r"[A-Z][a-z0-9]*", class_name)]


def _is_word_subsequence(tag_words: list[str], class_words: list[str]) -> bool:
    """Report whether every tag word appears in the class words, in order."""
    remaining = list(class_words)
    for word in tag_words:
        if word not in remaining:
            return False
        remaining = remaining[remaining.index(word) + 1 :]
    return True


_TAGGED_UNIONS = _tagged_unions()


def test_discovery_finds_the_known_unions_with_their_variants_intact() -> None:
    """Discovery finds known unions with multiple variants."""
    assert {"Message", "ContentPart", "TurnPart", "ResponseOutcome", "GenerateResult"} <= set(
        _TAGGED_UNIONS
    )
    undersized = {
        name: [variant.__name__ for variant in variants]
        for name, variants in _TAGGED_UNIONS.items()
        if len(variants) < 2
    }
    assert not undersized


def test_no_union_gives_two_variants_the_same_tag() -> None:
    """Each kind value identifies one variant of a tagged union."""
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


def test_generation_error_kind_matches_the_error_record_variants() -> None:
    """GenerationErrorKind contains each GenerationErrorRecord discriminator."""
    alias_values = set(get_args(GenerationErrorKind.__value__))
    record_values = {_tag_of(variant) for variant in _TAGGED_UNIONS["GenerationErrorRecord"]}
    assert alias_values == record_values
