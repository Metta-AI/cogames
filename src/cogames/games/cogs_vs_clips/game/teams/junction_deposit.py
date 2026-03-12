"""Junction deposit: aligned junctions receive resources and forward to team hub."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from cogames.core import CoGameMissionVariant, Deps
from cogames.games.cogs_vs_clips.game.elements import ElementsVariant
from cogames.games.cogs_vs_clips.game.teams.junction import TeamJunctionVariant
from cogames.games.cogs_vs_clips.game.teams.team import TeamVariant
from mettagrid.config.filter import actorHasAnyOf, hasTag, sharedTagPrefix
from mettagrid.config.handler_config import Handler, queryDeposit
from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.config.query import query
from mettagrid.config.tag import typeTag

if TYPE_CHECKING:
    from cogames.games.cogs_vs_clips.missions.mission import CvCMission


class JunctionDepositVariant(CoGameMissionVariant):
    """Allow depositing elements at aligned junctions into the matching team hub."""

    name: str = "junction_deposit"
    description: str = "Aligned junctions can deposit elements to the team's hub."

    @override
    def dependencies(self) -> Deps:
        return Deps(required=[TeamJunctionVariant, ElementsVariant])

    @override
    def modify_env(self, mission: CvCMission, env: MettaGridConfig) -> None:
        teams = list(mission.required_variant(TeamVariant).teams.values())
        elements = mission.required_variant(ElementsVariant).elements
        handlers: dict[str, Handler] = {}
        for t in teams:
            handlers[f"deposit_{t.name}"] = Handler(
                filters=[
                    hasTag(t.team_tag()),
                    sharedTagPrefix("team:"),
                    actorHasAnyOf(elements),
                ],
                mutations=[
                    queryDeposit(
                        query(typeTag("hub"), hasTag(t.team_tag())),
                        {resource: 100 for resource in elements},
                        stat_prefix=f"{t.name}/",
                    ),
                ],
            )
        env.game.objects["junction"].on_use_handlers.update(handlers)
