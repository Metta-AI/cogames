"""Timing stress variants for Overcogged cook and burn loops."""

from __future__ import annotations

from cogames.core import CoGameMissionVariant, Deps
from cogames.games.overcogged.defaults import FRIES_BURN_TICKS, FRIES_COOK_TICKS, SOUP_BURN_TICKS, SOUP_COOK_TICKS
from cogames.games.overcogged.variants.common import VariantGraphAccess
from cogames.games.overcogged.variants.mechanics import FullMechanicsVariant

SHORT_SOUP_COOK_TICKS = max(1, SOUP_COOK_TICKS // 2)
SHORT_FRIES_COOK_TICKS = max(1, FRIES_COOK_TICKS // 2)
LONG_SOUP_COOK_TICKS = SOUP_COOK_TICKS * 2
LONG_FRIES_COOK_TICKS = FRIES_COOK_TICKS * 2
FAST_SOUP_BURN_TICKS = max(1, SOUP_BURN_TICKS // 2)
FAST_FRIES_BURN_TICKS = max(1, FRIES_BURN_TICKS // 2)


class ShortCookVariant(CoGameMissionVariant):
    name: str = "short_cook"
    description: str = "Speed up soup and fries cook timers for mechanic stress tests."

    def dependencies(self) -> Deps:
        return Deps(required=[FullMechanicsVariant])

    def configure(self, deps: VariantGraphAccess) -> None:
        deps.required(FullMechanicsVariant)

    def modify_mission(self, mission) -> None:
        mission.soup_cook_ticks = SHORT_SOUP_COOK_TICKS
        mission.fries_cook_ticks = SHORT_FRIES_COOK_TICKS


class LongCookVariant(CoGameMissionVariant):
    name: str = "long_cook"
    description: str = "Slow down soup and fries cook timers for mechanic stress tests."

    def dependencies(self) -> Deps:
        return Deps(required=[FullMechanicsVariant])

    def configure(self, deps: VariantGraphAccess) -> None:
        deps.required(FullMechanicsVariant)

    def modify_mission(self, mission) -> None:
        mission.soup_cook_ticks = LONG_SOUP_COOK_TICKS
        mission.fries_cook_ticks = LONG_FRIES_COOK_TICKS


class FastBurnVariant(CoGameMissionVariant):
    name: str = "fast_burn"
    description: str = "Tighten soup and fries burn windows for readiness stress tests."

    def dependencies(self) -> Deps:
        return Deps(required=[FullMechanicsVariant])

    def configure(self, deps: VariantGraphAccess) -> None:
        deps.required(FullMechanicsVariant)

    def modify_mission(self, mission) -> None:
        mission.soup_burn_ticks = FAST_SOUP_BURN_TICKS
        mission.fries_burn_ticks = FAST_FRIES_BURN_TICKS
