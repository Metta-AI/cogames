"""Difficulty variants for Overcogged."""

from __future__ import annotations

from cogames.core import CoGameMissionVariant, Deps
from cogames.games.overcogged.variants.common import VariantGraphAccess
from cogames.games.overcogged.variants.layout import LayoutLineVariant
from cogames.games.overcogged.variants.mechanics import FullMechanicsVariant, SaladRecipeVariant


class EasyVariant(CoGameMissionVariant):
    name: str = "easy"
    description: str = "Lower ticket pressure, generous deadlines, slower burn."

    def dependencies(self) -> Deps:
        return Deps(required=[FullMechanicsVariant])

    def configure(self, deps: VariantGraphAccess) -> None:
        deps.required(FullMechanicsVariant)

    def modify_mission(self, mission) -> None:
        mission.ticket_interarrival = 24
        mission.ticket_deadline = 72
        mission.soup_cook_ticks = 8
        mission.soup_burn_ticks = 20
        mission.fries_cook_ticks = 6
        mission.fries_burn_ticks = 18
        mission.order_queue_max = 10
        mission.randomize_spawn_positions = False


class TutorialVariant(CoGameMissionVariant):
    name: str = "tutorial"
    description: str = "Low-pressure salad-first onboarding with wide line layout."

    def dependencies(self) -> Deps:
        return Deps(required=[SaladRecipeVariant, LayoutLineVariant])

    def configure(self, deps: VariantGraphAccess) -> None:
        deps.required(SaladRecipeVariant)
        deps.required(LayoutLineVariant)

    def modify_mission(self, mission) -> None:
        mission.ticket_interarrival = 30
        mission.ticket_deadline = 90
        mission.chop_ticks = 2
        mission.soup_cook_ticks = 7
        mission.soup_burn_ticks = 24
        mission.fries_cook_ticks = 5
        mission.fries_burn_ticks = 20
        mission.order_queue_max = 6
        mission.enable_soup_recipe = False
        mission.enable_fries_recipe = False
        mission.enable_wash_cycle = False
        mission.enable_soup_burn = False
        mission.enable_fries_burn = False


class HardVariant(CoGameMissionVariant):
    name: str = "hard"
    description: str = "Higher ticket pressure and shorter deadlines."

    def dependencies(self) -> Deps:
        return Deps(required=[FullMechanicsVariant])

    def configure(self, deps: VariantGraphAccess) -> None:
        deps.required(FullMechanicsVariant)

    def modify_mission(self, mission) -> None:
        mission.ticket_interarrival = 12
        mission.ticket_deadline = 36
        mission.soup_cook_ticks = 12
        mission.soup_burn_ticks = 10
        mission.fries_cook_ticks = 9
        mission.fries_burn_ticks = 8
        mission.order_queue_max = 8


class RushHourVariant(CoGameMissionVariant):
    name: str = "rush_hour"
    description: str = "Very high order pressure for stress testing."

    def dependencies(self) -> Deps:
        return Deps(required=[HardVariant])

    def configure(self, deps: VariantGraphAccess) -> None:
        deps.required(HardVariant)

    def modify_mission(self, mission) -> None:
        mission.ticket_interarrival = 9
        mission.ticket_deadline = 30
        mission.order_queue_max = 12
        mission.soup_burn_ticks = 8
        mission.fries_burn_ticks = 6
