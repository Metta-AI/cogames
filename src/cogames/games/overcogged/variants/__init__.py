"""Overcogged variants.

Public variants stay focused on launch-shipping knobs (difficulty/layout).
Mechanics variants are intentionally hidden from default listings but remain
available for curriculum/tree composition and explicit CLI usage.
"""

from __future__ import annotations

from typing import Sequence

from cogames.core import CoGameMissionVariant
from cogames.games.overcogged.variants.difficulty import EasyVariant, HardVariant, RushHourVariant, TutorialVariant
from cogames.games.overcogged.variants.layout import (
    LayoutCompactVariant,
    LayoutLineVariant,
    LayoutReverseVariant,
    LayoutTightFlowVariant,
    TightHubVariant,
)
from cogames.games.overcogged.variants.mechanics import (
    DishwashingVariant,
    FriesBurnVariant,
    FriesRecipeVariant,
    FullMechanicsVariant,
    QueueOrdersVariant,
    ResetMechanicsVariant,
    SaladRecipeVariant,
    SoupBurnVariant,
    SoupRecipeVariant,
)
from cogames.games.overcogged.variants.timing import FastBurnVariant, LongCookVariant, ShortCookVariant
from cogames.variants import VariantRegistry

PUBLIC_VARIANT_TYPES: tuple[type[CoGameMissionVariant], ...] = (
    TutorialVariant,
    EasyVariant,
    HardVariant,
    RushHourVariant,
    TightHubVariant,
    LayoutLineVariant,
    LayoutReverseVariant,
    LayoutCompactVariant,
    LayoutTightFlowVariant,
)
HIDDEN_VARIANT_TYPES: tuple[type[CoGameMissionVariant], ...] = (
    ResetMechanicsVariant,
    QueueOrdersVariant,
    SaladRecipeVariant,
    SoupRecipeVariant,
    DishwashingVariant,
    FriesRecipeVariant,
    SoupBurnVariant,
    FriesBurnVariant,
    ShortCookVariant,
    LongCookVariant,
    FastBurnVariant,
    FullMechanicsVariant,
)
ALL_VARIANT_TYPES = PUBLIC_VARIANT_TYPES + HIDDEN_VARIANT_TYPES

VARIANTS: tuple[CoGameMissionVariant, ...] = tuple(variant_type() for variant_type in PUBLIC_VARIANT_TYPES)
HIDDEN_VARIANTS: tuple[CoGameMissionVariant, ...] = tuple(variant_type() for variant_type in HIDDEN_VARIANT_TYPES)

_VARIANT_TYPES_BY_NAME = {variant.name: type(variant) for variant in [*VARIANTS, *HIDDEN_VARIANTS]}
HIDDEN_VARIANT_NAMES = frozenset(variant.name for variant in HIDDEN_VARIANTS)
_NON_MECHANIC_VARIANTS = {"reset_mechanics", "queue_orders", "short_cook", "long_cook", "fast_burn", "full"}
OVERCOOKED_MECHANICS: tuple[str, ...] = tuple(
    variant.name for variant in HIDDEN_VARIANTS if variant.name not in _NON_MECHANIC_VARIANTS
)

OVERCOOKED_INTERFACE_VARIANTS: tuple[str, ...] = ("full",)


def normalize_variant_names(names: Sequence[str]) -> list[str]:
    requested: list[str] = []
    seen: set[str] = set()
    for raw_name in names:
        name = raw_name.strip()
        if not name or name in seen:
            continue
        seen.add(name)
        requested.append(name)
    unknown = [name for name in requested if name not in _VARIANT_TYPES_BY_NAME]
    if unknown:
        available = ", ".join(sorted(_VARIANT_TYPES_BY_NAME))
        raise ValueError(f"Unknown variant {unknown[0]!r}. Available: {available}")
    return requested


def _instantiate_variant(name: str) -> CoGameMissionVariant:
    expected_type = _VARIANT_TYPES_BY_NAME[name]
    return expected_type()  # pyright: ignore[reportCallIssue]


def resolve_variant_selection(names: Sequence[str]) -> VariantRegistry:
    requested_names = normalize_variant_names(names)
    registry = VariantRegistry([_instantiate_variant(name) for name in requested_names])
    registry.run_configure(requested_names)

    unexpected = [variant.name for variant in registry.all() if not isinstance(variant, ALL_VARIANT_TYPES)]
    assert not unexpected, f"Overcogged registry resolved non-local variants: {unexpected}"
    return registry


def parse_variants(names: Sequence[str]) -> list[CoGameMissionVariant]:
    return list(resolve_variant_selection(names).configured())
