"""The shape of the kind tag, checked structurally over every tagged union langchaint defines.

The unions are discovered by importing every langchaint module and reading its type aliases, so a
union added anywhere is covered the moment it exists. A tagged union is one whose members all
annotate kind, so a union with a builtin member, which cannot hold a tag, is out of scope rather
than a failure.

Two invariants hold whatever the union: two members never share a tag, and every word of a tag comes
from its own class's name. Together they are what lets a match on kind route each member to its own
case.
The naming check is a subsequence test because a member drops the words its siblings all share
(UserMessage tags "user"), and a member of two unions carries one tag under both, so the words
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
        return [member for argument in get_args(annotation) for member in _flatten_union(argument)]
    origin = get_origin(annotation)
    if isinstance(origin, type):
        return [origin]
    return [annotation] if isinstance(annotation, type) else []


def _own_annotations(member: type) -> dict[str, object]:
    """Read the annotations the class declares itself, ignoring any it inherits.

    inspect.get_annotations reads them wherever the interpreter keeps them, and evaluates any it
    finds deferred. A 3.13 class dict holds __annotations__; a 3.14 one holds __annotate_func__ and
    caches under __annotations_cache__, so reading __annotations__ out of the dict finds nothing.
    """
    return inspect.get_annotations(member)


def _tagged_unions() -> dict[str, tuple[type, ...]]:
    """Map each langchaint type alias whose members all declare kind to those members."""
    found: dict[str, tuple[type, ...]] = {}
    for module in package_modules():
        for attribute_name, value in vars(module).items():
            if getattr(value, "__value__", None) is None:
                continue
            members = _flatten_union(value)
            if members and all("kind" in _own_annotations(member) for member in members):
                found[attribute_name] = tuple(members)
    return found


def _tag_of(member: type) -> str:
    """Read the one string a member's kind Literal holds."""
    literal_arguments = get_args(_own_annotations(member)["kind"])
    assert len(literal_arguments) == 1, f"{member.__name__}.kind holds no single Literal value"
    tag = literal_arguments[0]
    assert isinstance(tag, str), f"{member.__name__}.kind is not a string Literal"
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


def test_discovery_finds_the_known_unions_with_their_members_intact() -> None:
    """Discovery reaches both the pydantic unions and the outcome unions, and every member of each.

    Every check below passes vacuously on a union discovery dropped, and passes just as quietly on
    one it found but flattened to a single member, so a walk that stops early has to fail here.
    A union of one member is not a union, which is what makes the lower bound safe to assert.
    """
    assert {"Message", "Part", "TurnElement", "ResponseOutcome"} <= set(_TAGGED_UNIONS)
    undersized = {
        name: [member.__name__ for member in members]
        for name, members in _TAGGED_UNIONS.items()
        if len(members) < 2
    }
    assert not undersized


def test_no_union_gives_two_members_the_same_tag() -> None:
    """Within one union every tag is distinct, so a tag identifies one member.

    A member that copied a sibling's tag would take the sibling's case in every match on kind,
    leaving its own case dead, and pyrefly reports neither the duplicate nor the dead case.
    """
    collisions = {}
    for union_name, members in _TAGGED_UNIONS.items():
        tags = [_tag_of(member) for member in members]
        duplicated = sorted({tag for tag in tags if tags.count(tag) > 1})
        if duplicated:
            collisions[union_name] = duplicated
    assert not collisions


def test_every_tag_is_built_from_its_own_class_name() -> None:
    """A tag's words appear in its class's name, in order."""
    members = {member for union_members in _TAGGED_UNIONS.values() for member in union_members}
    misnamed = sorted(
        (member.__name__, _tag_of(member))
        for member in members
        if not _is_word_subsequence(_tag_of(member).split("_"), _words(member.__name__))
    )
    assert not misnamed
