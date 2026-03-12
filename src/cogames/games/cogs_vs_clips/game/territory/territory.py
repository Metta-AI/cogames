"""Territory variant: adds team territory, junction network, and territory control."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from pydantic import Field

from cogames.core import CoGameMissionVariant, Deps
from cogames.games.cogs_vs_clips.game.clips.clips import ClipsVariant
from cogames.games.cogs_vs_clips.game.junction import JunctionVariant
from cogames.games.cogs_vs_clips.game.teams.hub import CvCHubConfig, TeamHubVariant
from cogames.games.cogs_vs_clips.game.teams.team import TeamVariant
from mettagrid.config.filter import hasTag, maxDistance, sharedTagPrefix
from mettagrid.config.handler_config import Handler, updateTarget
from mettagrid.config.mettagrid_config import GridObjectConfig, MettaGridConfig
from mettagrid.config.query import ClosureQuery, MaterializedQuery, materializedQuery, query
from mettagrid.config.territory_config import TerritoryConfig, TerritoryControlConfig

JUNCTION_ALIGN_DISTANCE = 15
HUB_ALIGN_DISTANCE = 25
TERRITORY_CONTROL_RADIUS = 10

if TYPE_CHECKING:
    from cogames.games.cogs_vs_clips.game.teams import TeamConfig
    from cogames.games.cogs_vs_clips.missions.mission import CvCMission


def net_materialized_query(team: TeamConfig) -> MaterializedQuery:
    """Build the closure query that computes which junctions are in a team's network."""
    net_tag = f"net:{team.name}"
    source = getattr(team, "ship_query", lambda: CvCHubConfig.hub_query(team))()
    return materializedQuery(
        net_tag,
        ClosureQuery(
            source=source,
            candidates=query("type:junction", hasTag(team.team_tag())),
            edge_filters=[maxDistance(max(JUNCTION_ALIGN_DISTANCE, HUB_ALIGN_DISTANCE))],
        ),
    )


class TerritoryVariant(CoGameMissionVariant):
    """Add team territory with junction-based influence, network queries, and alignment stats."""

    name: str = "territory"
    description: str = "Team territory with junction network and alignment tracking."

    control_range: int = Field(default=TERRITORY_CONTROL_RADIUS)

    @override
    def dependencies(self) -> Deps:
        return Deps(required=[TeamHubVariant, JunctionVariant, TeamVariant], optional=[ClipsVariant])

    @override
    def modify_env(self, mission: CvCMission, env: MettaGridConfig) -> None:
        presence: dict[str, Handler] = {}
        if mission.has_variant("energy"):
            presence["heal"] = Handler(
                filters=[sharedTagPrefix("team:")],
                mutations=[updateTarget({"energy": 100})],
            )

        env.game.territories["team_territory"] = TerritoryConfig(
            tag_prefix="team:",
            presence=presence,
        )

        tc = TerritoryControlConfig(territory="team_territory", strength=self.control_range)

        for obj in env.game.objects.values():
            if isinstance(obj, GridObjectConfig) and obj.name in ("hub", "junction", "ship"):
                obj.territory_controls.append(tc.model_copy())

        team_v = mission.required_variant(TeamVariant)
        clips_v = mission.optional_variant(ClipsVariant)
        all_teams = list(team_v.teams.values())
        if clips_v is not None and clips_v.clips is not None:
            all_teams.append(clips_v.clips)

        for team in all_teams:
            env.game.materialize_queries.append(net_materialized_query(team))
