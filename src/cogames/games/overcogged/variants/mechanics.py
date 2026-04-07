"""Mechanics variants for Overcogged tree composition."""

from __future__ import annotations

from cogames.core import CoGameMissionVariant, Deps
from cogames.games.overcogged.variants.common import VariantGraphAccess


class ResetMechanicsVariant(CoGameMissionVariant):
    name: str = "reset_mechanics"
    description: str = "Reset all kitchen mechanics to off. Used as a dependency anchor."

    def modify_mission(self, mission) -> None:
        mission.enable_queue_orders = False
        mission.enable_salad_recipe = False
        mission.enable_soup_recipe = False
        mission.enable_fries_recipe = False
        mission.enable_wash_cycle = False
        mission.enable_soup_burn = False
        mission.enable_fries_burn = False


class QueueOrdersVariant(CoGameMissionVariant):
    name: str = "queue_orders"
    description: str = "Enable ticket arrivals/expiry and active order queue pressure."

    def dependencies(self) -> Deps:
        return Deps(required=[ResetMechanicsVariant])

    def configure(self, deps: VariantGraphAccess) -> None:
        deps.required(ResetMechanicsVariant)

    def modify_mission(self, mission) -> None:
        mission.enable_queue_orders = True


class SaladRecipeVariant(CoGameMissionVariant):
    name: str = "salad_recipe"
    description: str = "Enable salad prep -> plate -> serve mechanics."

    def dependencies(self) -> Deps:
        return Deps(required=[QueueOrdersVariant])

    def configure(self, deps: VariantGraphAccess) -> None:
        deps.required(QueueOrdersVariant)

    def modify_mission(self, mission) -> None:
        mission.enable_salad_recipe = True


class SoupRecipeVariant(CoGameMissionVariant):
    name: str = "soup_recipe"
    description: str = "Enable soup prep/cook/serve mechanics."

    def dependencies(self) -> Deps:
        return Deps(required=[QueueOrdersVariant])

    def configure(self, deps: VariantGraphAccess) -> None:
        deps.required(QueueOrdersVariant)

    def modify_mission(self, mission) -> None:
        mission.enable_soup_recipe = True


class FriesRecipeVariant(CoGameMissionVariant):
    name: str = "fries_recipe"
    description: str = "Enable fries prep/fry/serve mechanics."

    def dependencies(self) -> Deps:
        return Deps(required=[QueueOrdersVariant])

    def configure(self, deps: VariantGraphAccess) -> None:
        deps.required(QueueOrdersVariant)

    def modify_mission(self, mission) -> None:
        mission.enable_fries_recipe = True


class DishwashingVariant(CoGameMissionVariant):
    name: str = "dishwashing"
    description: str = "Enable dirty-plate -> clean-plate wash loop."

    def dependencies(self) -> Deps:
        return Deps(required=[QueueOrdersVariant])

    def configure(self, deps: VariantGraphAccess) -> None:
        deps.required(QueueOrdersVariant)

    def modify_mission(self, mission) -> None:
        mission.enable_wash_cycle = True


class SoupBurnVariant(CoGameMissionVariant):
    name: str = "soup_burn"
    description: str = "Enable soup burn hazard and pot clear loop."

    def dependencies(self) -> Deps:
        return Deps(required=[SoupRecipeVariant])

    def configure(self, deps: VariantGraphAccess) -> None:
        deps.required(SoupRecipeVariant)

    def modify_mission(self, mission) -> None:
        mission.enable_soup_burn = True


class FriesBurnVariant(CoGameMissionVariant):
    name: str = "fries_burn"
    description: str = "Enable fries burn hazard and fryer clear loop."

    def dependencies(self) -> Deps:
        return Deps(required=[FriesRecipeVariant])

    def configure(self, deps: VariantGraphAccess) -> None:
        deps.required(FriesRecipeVariant)

    def modify_mission(self, mission) -> None:
        mission.enable_fries_burn = True


class FullMechanicsVariant(CoGameMissionVariant):
    name: str = "full"
    description: str = "Enable full Overcogged launch mechanics."

    def dependencies(self) -> Deps:
        return Deps(
            required=[
                SaladRecipeVariant,
                SoupRecipeVariant,
                DishwashingVariant,
                FriesRecipeVariant,
                SoupBurnVariant,
                FriesBurnVariant,
            ]
        )

    def configure(self, deps: VariantGraphAccess) -> None:
        deps.required(SaladRecipeVariant)
        deps.required(SoupRecipeVariant)
        deps.required(DishwashingVariant)
        deps.required(FriesRecipeVariant)
        deps.required(SoupBurnVariant)
        deps.required(FriesBurnVariant)
