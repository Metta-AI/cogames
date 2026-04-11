"""Gear stations variant: creates universal gear stations."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from pydantic import Field

from cogames.core import CoGameMissionVariant, Deps
from cogames.games.cogs_vs_clips.game.gear import GearVariant
from cogames.games.cogs_vs_clips.game.terrain import BuildingsVariant
from cogames.variants import ResolvedDeps
from mettagrid.config.handler_config import (
    ClearInventoryMutation,
    EntityTarget,
    Handler,
    actorHas,
    firstMatch,
    updateActor,
)
from mettagrid.config.mettagrid_config import GridObjectConfig, MettaGridConfig

if TYPE_CHECKING:
    from cogames.games.cogs_vs_clips.missions.mission import CvCMission


class GearStationsVariant(CoGameMissionVariant):
    """Create universal gear stations that charge the agent directly."""

    name: str = "gear_stations"
    description: str = "Place gear stations on the map."

    @override
    def dependencies(self) -> Deps:
        return Deps(required=[GearVariant])

    @override
    def modify_env(self, mission: CvCMission, env: MettaGridConfig) -> None:
        gear = mission.required_variant(GearVariant)
        for item_name in gear.items:
            cost = gear.station_costs.get(item_name, {})
            station = env.game.objects[item_name] = GridObjectConfig(name=item_name)
            symbol = gear.station_symbols.get(item_name)
            if symbol is not None:
                env.game.render.symbols[item_name] = symbol
            station.on_use_handler = firstMatch(
                [
                    Handler(name="keep_gear", filters=[actorHas({item_name: 1})], mutations=[]),
                    Handler(
                        name="change_gear",
                        filters=[actorHas(cost)] if cost else [],
                        mutations=[
                            ClearInventoryMutation(target=EntityTarget.ACTOR, limit_name="gear"),
                            updateActor({k: -v for k, v in cost.items()}),
                            updateActor({item_name: 1}),
                        ],
                    ),
                ]
            )


class WildGearStationsVariant(CoGameMissionVariant):
    """Scatter gear stations across the map as buildings."""

    name: str = "wild_gear_stations"
    description: str = "Place gear stations randomly across the map."
    density: float = Field(default=0.1, description="Building density for each gear station type.")

    @override
    def dependencies(self) -> Deps:
        return Deps(required=[GearStationsVariant, BuildingsVariant, GearVariant])

    @override
    def configure(self, deps: ResolvedDeps) -> None:
        deps.required(GearStationsVariant)
        terrain = deps.required(BuildingsVariant)
        for item_name in deps.required(GearVariant).items:
            terrain.building_density.setdefault(item_name, self.density)
