"""Heal-team variant: territory heals energy and HP for team members."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from cogames.core import CoGameMissionVariant, Deps
from cogames.games.cogs_vs_clips.game.damage import DamageVariant
from cogames.games.cogs_vs_clips.game.energy import EnergyVariant
from cogames.games.cogs_vs_clips.game.territory.territory import TerritoryVariant
from mettagrid.config.filter import sharedTagPrefix
from mettagrid.config.handler_config import Handler, updateTarget
from mettagrid.config.mettagrid_config import MettaGridConfig

if TYPE_CHECKING:
    from cogames.games.cogs_vs_clips.missions.mission import CvCMission

ENERGY_HEAL_RATE = 100
HP_HEAL_RATE = 100


class HealTeamVariant(CoGameMissionVariant):
    """Territory heals team members' energy and HP each tick they remain inside."""

    name: str = "heal_team"
    description: str = "Territory heals energy and HP for team members."

    energy_heal_rate: int = ENERGY_HEAL_RATE
    hp_heal_rate: int = HP_HEAL_RATE

    @override
    def dependencies(self) -> Deps:
        return Deps(required=[TerritoryVariant, EnergyVariant], optional=[DamageVariant])

    @override
    def modify_env(self, mission: CvCMission, env: MettaGridConfig) -> None:
        territory = env.game.territories.get("team_territory")
        if territory is None:
            return

        if self.energy_heal_rate and "energy" in env.game.resource_names:
            territory.presence["heal_energy"] = Handler(
                filters=[sharedTagPrefix("team:")],
                mutations=[updateTarget({"energy": self.energy_heal_rate})],
            )

        if self.hp_heal_rate and "hp" in env.game.resource_names:
            territory.presence["heal_hp"] = Handler(
                filters=[sharedTagPrefix("team:")],
                mutations=[updateTarget({"hp": self.hp_heal_rate})],
            )
