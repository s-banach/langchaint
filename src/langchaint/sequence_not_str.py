"""Define a sequence protocol that rejects bare `str` values."""

from collections.abc import Iterator, Sequence
from typing import Protocol, SupportsIndex, overload


class SequenceNotStr[T_co](Protocol):
    """A `Sequence` whose structural type rejects bare `str` values.

    It matches `list` and `tuple` while retaining covariance.
    Its runtime users still reject bare `str` values explicitly.
    """

    @overload
    def __getitem__(self, index: SupportsIndex, /) -> T_co: ...

    @overload
    def __getitem__(self, index: slice, /) -> Sequence[T_co]: ...

    def __contains__(self, value: object, /) -> bool:
        """Accept any object for membership testing."""
        ...

    def __len__(self) -> int:
        """Return the sequence length."""
        ...

    def __iter__(self) -> Iterator[T_co]:
        """Iterate over the sequence."""
        ...

    def __reversed__(self) -> Iterator[T_co]:
        """Iterate over the sequence in reverse."""
        ...
