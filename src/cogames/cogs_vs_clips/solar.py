"""Solar variant: agent solar inventory and solar-to-energy conversion."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from pydantic import Field

from cogames.cogs_vs_clips.energy import EnergyVariant
from cogames.core import CoGameMissionVariant, Deps
from cogames.variants import ResolvedDeps
from mettagrid.config.game_value import inv
from mettagrid.config.handler_config import EntityTarget, Handler
from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.config.mutation.game_value_mutation import SetGameValueMutation

if TYPE_CHECKING:
    from cogames.core import CoGameMission


class SolarVariant(CoGameMissionVariant):
    name: str = "solar"

    initial_solar: int = Field(default=1)

    @override
    def dependencies(self) -> Deps:
        return Deps(required=[EnergyVariant])

    @override
    def configure(self, deps: ResolvedDeps) -> None:
        deps.required(EnergyVariant)

    @override
    def modify_env(self, mission: CoGameMission, env: MettaGridConfig) -> None:
        if "solar" not in env.game.resource_names:
            env.game.resource_names.append("solar")
        for agent in env.game.agents:
            agent.inventory.initial["solar"] = self.initial_solar
            agent.on_tick["solar_to_energy"] = Handler(
                mutations=[SetGameValueMutation(value=inv("energy"), source=inv("solar"), target=EntityTarget.ACTOR)]
            )
