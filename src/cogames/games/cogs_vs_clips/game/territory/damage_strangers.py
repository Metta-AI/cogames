"""Damage-strangers variant: territory attacks non-team members."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from cogames.core import CoGameMissionVariant, Deps
from cogames.games.cogs_vs_clips.game.damage import DamageVariant
from cogames.games.cogs_vs_clips.game.territory.territory import TerritoryVariant
from mettagrid.config.filter import isNot, sharedTagPrefix
from mettagrid.config.handler_config import Handler, updateTarget
from mettagrid.config.mettagrid_config import MettaGridConfig

if TYPE_CHECKING:
    from cogames.games.cogs_vs_clips.missions.mission import CvCMission

DAMAGE_RATE = -1


class DamageStrangersVariant(CoGameMissionVariant):
    """Territory damages HP of non-team agents each tick they remain inside."""

    name: str = "damage_strangers"
    description: str = "Territory damages HP of non-team members."

    damage_rate: int = DAMAGE_RATE

    @override
    def dependencies(self) -> Deps:
        return Deps(required=[TerritoryVariant, DamageVariant])

    @override
    def modify_env(self, mission: CvCMission, env: MettaGridConfig) -> None:
        territory = env.game.territories.get("team_territory")
        if territory is None:
            return

        territory.presence["damage_strangers"] = Handler(
            filters=[isNot(sharedTagPrefix("team:"))],
            mutations=[updateTarget({"hp": self.damage_rate})],
        )
