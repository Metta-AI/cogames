from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from pydantic import Field

from cogames.cogs_vs_clips.config import CvCConfig
from cogames.cogs_vs_clips.stations import CvCStationConfig
from mettagrid.config.filter import actorHasAnyOf, actorHasTag, hasTagPrefix, sharedTagPrefix
from mettagrid.config.handler_config import (
    AOEConfig,
    Handler,
    actorHas,
    collectiveDeposit,
    updateActor,
    updateTarget,
)
from mettagrid.config.mettagrid_config import GridObjectConfig
from mettagrid.config.mutation import recomputeMaterializedQuery, removeTag, removeTagPrefix
from mettagrid.config.mutation.alignment_mutation import removeAlignment
from mettagrid.config.mutation.stats_mutation import logActorAgentStat

if TYPE_CHECKING:
    from cogames.cogs_vs_clips.team import TeamConfig


def _neg(recipe: dict[str, int]) -> dict[str, int]:
    return {k: -v for k, v in recipe.items()}


class CvCJunctionConfig(CvCStationConfig):
    """Supply depot that receives element resources via default vibe into collective."""

    aoe_range: int = Field(default=CvCConfig.JUNCTION_DISTANCE, description="Range for AOE effects")
    heal_deltas: dict[str, int] = Field(default_factory=lambda: {"energy": 100, "hp": 100})
    attack_deltas: dict[str, int] = Field(default_factory=lambda: {"hp": -1})

    def station_cfg(
        self,
        teams: list[TeamConfig],
        owner_team_name: Optional[str] = None,
        map_name: Optional[str] = None,
    ) -> GridObjectConfig:
        return GridObjectConfig(
            name="junction",
            map_name=map_name or "junction",
            render_name="junction",
            render_symbol="📦",
            tags=[
                f"team:{owner_team_name}",
            ]
            if owner_team_name
            else [],
            collective=owner_team_name,
            on_tag_remove={
                f"net:{t.name}": Handler(
                    filters=[],
                    mutations=[
                        removeTag(f"team:{t.name}"),
                        removeTag(f"collective:{t.name}"),
                        removeAlignment(),
                    ],
                )
                for t in teams
            },
            aoes={
                "base": AOEConfig(
                    radius=self.aoe_range,
                    filters=[sharedTagPrefix("team:")],
                    mutations=[updateTarget(self.heal_deltas)],
                    presence_deltas={"influence": 1},
                    controls_territory=True,
                ),
            },
            on_use_handlers={
                **{
                    f"deposit_{t.name}": Handler(
                        filters=[
                            actorHasTag(t.team_tag()),
                            sharedTagPrefix("team:"),
                            actorHasAnyOf(CvCConfig.ELEMENTS),
                        ],
                        mutations=[
                            collectiveDeposit({resource: 100 for resource in CvCConfig.ELEMENTS}),
                        ],
                    )
                    for t in teams
                },
                "scramble": Handler(
                    filters=[
                        hasTagPrefix("team:"),
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
