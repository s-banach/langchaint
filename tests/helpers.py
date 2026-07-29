"""Helpers shared by more than one test module.

A helper lands here when a second module needs it; one used by a single module stays in that module.
"""

import importlib
import math
import pkgutil
from collections.abc import Iterator
from types import ModuleType

from pydantic import BaseModel

import langchaint
from langchaint import Billing, Usage


def stated_billing(
    usage: Usage,
    *,
    input_cache_none_usd_per_million_tokens: float = math.nan,
    usage_raw: BaseModel | None = None,
) -> Billing:
    """Carry a Usage a test stated outright as the Billing an attempt record holds.

    The tier is filler and every rate defaults to NaN: a test that states its costs on the Usage is
    exercising what the tree does with a Billing, not how a rate table built one. A test of
    cache_savings_in_usd states the one rate that property reads.
    usage_raw defaults to None, the provider having reported no usage object.
    """
    return Billing(
        usage=usage,
        service_tier="stub",
        usage_raw=usage_raw,
        input_cache_none_usd_per_million_tokens=input_cache_none_usd_per_million_tokens,
        cache_read_usd_per_million_tokens=math.nan,
        cache_write_usd_per_million_tokens=math.nan,
        output_usd_per_million_tokens=math.nan,
    )


def package_modules() -> Iterator[ModuleType]:
    """Import every module under langchaint, backend subpackages included.

    Importing the backend subpackages and tracing requires both SDKs and opentelemetry-api,
    which the dev environment installs for the adapter and tracing tests.

    Yields:
        Each imported module, the package itself first.
    """
    yield langchaint
    for module_info in pkgutil.walk_packages(langchaint.__path__, prefix="langchaint."):
        yield importlib.import_module(module_info.name)


def uniform_returns_ceiling(_low: float, high: float) -> float:
    """Stand in for random.uniform, returning the ceiling of the range.

    Patched over the random.uniform rate_limiter draws its full jitter from, this makes a backoff delay its ceiling.
    A test can then state the delay the retry loop waits.
    """
    return high
