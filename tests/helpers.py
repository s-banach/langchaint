"""Helpers shared by more than one test module.

A helper lands here when a second module needs it; one used by a single module stays in that module.
"""


def uniform_returns_ceiling(_low: float, high: float) -> float:
    """Stand in for random.uniform, returning the ceiling of the range.

    Patched over the random.uniform rate_limiter draws its full jitter from, this makes a backoff delay its ceiling.
    A test can then state the delay the retry loop waits.
    """
    return high
