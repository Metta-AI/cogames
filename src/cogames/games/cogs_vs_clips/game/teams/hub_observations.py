"""Hub observations variant: adds per-team hub inventory to global observations."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from cogames.core import CoGameMissionVariant, Deps
from cogames.games.cogs_vs_clips.game.elements import ElementsVariant
from cogames.games.cogs_vs_clips.game.teams.hub import TeamHubVariant
from mettagrid.config.filter import sharedTagPrefix
from mettagrid.config.game_value import QueryInventoryValue
from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.config.query import query
from mettagrid.config.tag import typeTag

if TYPE_CHECKING:
    from cogames.games.cogs_vs_clips.missions.mission import CvCMission


class HubObservationsVariant(CoGameMissionVariant):
    """Add per-team hub resource amounts to each agent's global observations."""

    name: str = "hub_observations"
    description: str = "Agents observe their own team's hub inventory."

    @override
    def dependencies(self) -> Deps:
        return Deps(required=[TeamHubVariant, ElementsVariant])

    @override
    def modify_env(self, mission: CvCMission, env: MettaGridConfig) -> None:
        for resource in mission.required_variant(ElementsVariant).elements:
            env.game.obs.global_obs.obs[f"team:{resource}"] = QueryInventoryValue(
                query=query(typeTag("hub"), sharedTagPrefix("team:")),
                item=resource,
            )
