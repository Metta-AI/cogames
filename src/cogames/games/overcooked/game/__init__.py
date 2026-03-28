"""Lazy Overcooked variant exports for the canonical CoGames game package."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cogames.core import CoGameMissionVariant

    PUBLIC_VARIANTS: list[CoGameMissionVariant]
    HIDDEN_VARIANTS: list[CoGameMissionVariant]
    VARIANTS: list[CoGameMissionVariant]


def load_variants() -> tuple[list[CoGameMissionVariant], list[CoGameMissionVariant], list[CoGameMissionVariant]]:
    variants_module = importlib.import_module("metta.games.overcooked.variants")
    hidden_variants = variants_module.HIDDEN_VARIANTS
    public_variants = variants_module.VARIANTS

    return public_variants, hidden_variants, public_variants + hidden_variants


def __getattr__(name: str) -> object:
    public_variants, hidden_variants, variants = load_variants()
    values = {
        "PUBLIC_VARIANTS": public_variants,
        "HIDDEN_VARIANTS": hidden_variants,
        "VARIANTS": variants,
    }
    if name in values:
        return values[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["PUBLIC_VARIANTS", "HIDDEN_VARIANTS", "VARIANTS", "load_variants"]
