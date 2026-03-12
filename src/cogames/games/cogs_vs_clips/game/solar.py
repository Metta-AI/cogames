"""Solar variant: adds solar-to-energy conversion on each tick."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from cogames.core import CoGameMissionVariant, Deps
from cogames.games.cogs_vs_clips.game.energy import EnergyVariant
from cogames.variants import ResolvedDeps
from mettagrid.config.game_value import inv
from mettagrid.config.handler_config import EntityTarget, Handler
from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.config.mutation.game_value_mutation import SetGameValueMutation

if TYPE_CHECKING:
    from cogames.games.cogs_vs_clips.missions.mission import CvCMission


class SolarVariant(CoGameMissionVariant):
    """Add solar_to_energy on_tick handler so solar regenerates energy."""

    name: str = "solar"
    description: str = "Solar resource converts to energy each tick."

    @override
    def dependencies(self) -> Deps:
        return Deps(required=[EnergyVariant])

    @override
    def configure(self, deps: ResolvedDeps) -> None:
        deps.required(EnergyVariant)

    @override
    def modify_env(self, mission: CvCMission, env: MettaGridConfig) -> None:
        env.game.add_resource("solar")

        for agent in env.game.agents:
            agent.inventory.initial["solar"] = 1
            agent.on_tick["solar_to_energy"] = Handler(
                mutations=[SetGameValueMutation(value=inv("energy"), source=inv("solar"), target=EntityTarget.ACTOR)]
            )
