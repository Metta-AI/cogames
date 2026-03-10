"""Solar variant: agent solar inventory and solar-to-energy conversion."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from cogames.cogs_vs_clips.energy import EnergyVariant
from cogames.core import CoGameMissionVariant, Deps
from mettagrid.config.game_value import inv
from mettagrid.config.handler_config import EntityTarget, Handler
from mettagrid.config.mutation.game_value_mutation import SetGameValueMutation

if TYPE_CHECKING:
    from cogames.cogs_vs_clips.mission import CvCMission
    from mettagrid.config.mettagrid_config import MettaGridConfig


class SolarVariant(CoGameMissionVariant):
    name: str = "solar"

    @override
    def dependencies(self) -> Deps:
        return Deps(required=[EnergyVariant])

    @override
    def modify_env(self, mission: CvCMission, env: MettaGridConfig) -> None:
        cog = mission.cog
        for agent in env.game.agents:
            agent.inventory.initial["solar"] = cog.initial_solar
            agent.on_tick["solar_to_energy"] = Handler(
                mutations=[SetGameValueMutation(value=inv("energy"), source=inv("solar"), target=EntityTarget.ACTOR)]
            )
