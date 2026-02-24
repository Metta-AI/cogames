from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from pydantic import Field

from cogames.cogs_vs_clips.config import CvCConfig
from cogames.cogs_vs_clips.stations import CvCStationConfig
from mettagrid.config.filter import actorHasAnyOf, actorHasTag, hasTag, hasTagPrefix, isNot, sharedTagPrefix
from mettagrid.config.handler_config import (
    Handler,
    actorHas,
    queryDeposit,
    updateActor,
)
from mettagrid.config.mettagrid_config import GridObjectConfig
from mettagrid.config.mutation import recomputeMaterializedQuery, removeTag, removeTagPrefix
from mettagrid.config.mutation.stats_mutation import logActorAgentStat
from mettagrid.config.query import query
from mettagrid.config.tag import typeTag
from mettagrid.config.territory_config import TerritoryControlConfig

if TYPE_CHECKING:
    from cogames.cogs_vs_clips.team import TeamConfig


def _neg(recipe: dict[str, int]) -> dict[str, int]:
    return {k: -v for k, v in recipe.items()}


class CvCJunctionConfig(CvCStationConfig):
    """Junction station that can be aligned to a team via tags."""

    control_range: int = Field(default=CvCConfig.TERRITORY_CONTROL_RADIUS, description="Range for territory control")

    def station_cfg(
        self,
        teams: list[TeamConfig],
        owner_team_name: Optional[str] = None,
        map_name: Optional[str] = None,
    ) -> GridObjectConfig:
        return GridObjectConfig(
            name="junction",
            map_name=map_name or "junction",
            render_symbol="📦",
            tags=[
                f"team:{owner_team_name}",
            ]
            if owner_team_name
            else [],
            on_tag_remove={
                f"net:{t.name}": Handler(
                    filters=[],
                    mutations=[
                        removeTag(f"team:{t.name}"),
                    ],
                )
                for t in teams
            },
            territory_controls=[
                TerritoryControlConfig(territory="team_territory", strength=self.control_range),
            ],
            on_use_handlers={
                **{
                    f"deposit_{t.name}": Handler(
                        filters=[
                            actorHasTag(t.team_tag()),
                            sharedTagPrefix("team:"),
                            actorHasAnyOf(CvCConfig.ELEMENTS),
                        ],
                        mutations=[
                            queryDeposit(
                                query(typeTag("hub"), hasTag(t.team_tag())),
                                {resource: 100 for resource in CvCConfig.ELEMENTS},
                            ),
                        ],
                    )
                    for t in teams
                },
                "scramble": Handler(
                    filters=[
                        hasTagPrefix("team:"),
                        isNot(sharedTagPrefix("team:")),
                        actorHas({"scrambler": 1, **CvCConfig.SCRAMBLE_COST}),
                    ],
                    mutations=[
                        removeTagPrefix("net:"),
                        updateActor(_neg(CvCConfig.SCRAMBLE_COST)),
                        logActorAgentStat("junction.scrambled_by_agent"),
                        recomputeMaterializedQuery("net:"),
                    ],
                ),
                **{
                    f"align_{t.name}": Handler(
                        filters=[
                            actorHas({"aligner": 1, **CvCConfig.ALIGN_COST}),
                            *t.junction_is_alignable(),
                        ],
                        mutations=[
                            updateActor(_neg(CvCConfig.ALIGN_COST)),
                            logActorAgentStat("junction.aligned_by_agent"),
                            *t.junction_align_mutations(),
                        ],
                    )
                    for t in teams
                },
            },
        )
