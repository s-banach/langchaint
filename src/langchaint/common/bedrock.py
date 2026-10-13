"""Resolve Bedrock cross-region identifiers, without langchaint imports outside `common/`."""

from collections.abc import Container, Mapping
from typing import NamedTuple


class BedrockCatalogResolution(NamedTuple):
    """The catalog identifier a Bedrock identifier resolves through."""

    catalog_id: str
    cross_region_multiplier: float | None
    """The multiplier of a stripped cross-region prefix, or `None` when no prefix was stripped."""


def resolve_bedrock_catalog_id(
    model: str,
    *,
    catalog_ids: Container[str],
    cross_region_multipliers: Mapping[str, float],
) -> BedrockCatalogResolution:
    """Resolve a Bedrock identifier to its catalog identifier.

    An identifier in `catalog_ids` resolves to itself.
    Otherwise a cross-region prefix in `cross_region_multipliers` is stripped, keeping its multiplier.
    Any other identifier resolves to itself.
    """
    if model in catalog_ids:
        return BedrockCatalogResolution(catalog_id=model, cross_region_multiplier=None)
    prefix, _, catalog_id = model.partition(".")
    cross_region_multiplier = cross_region_multipliers.get(prefix)
    if cross_region_multiplier is None:
        return BedrockCatalogResolution(catalog_id=model, cross_region_multiplier=None)
    return BedrockCatalogResolution(
        catalog_id=catalog_id, cross_region_multiplier=cross_region_multiplier
    )
