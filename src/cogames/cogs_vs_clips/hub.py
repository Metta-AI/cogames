from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from pydantic import Field

from cogames.cogs_vs_clips.config import CvCConfig
from cogames.cogs_vs_clips.stations import CvCStationConfig, _neg, _opposing_team_filters
from mettagrid.config.filter import actorHasAnyOf, sharedTagPrefix
from mettagrid.config.handler_config import (
    AOEConfig,
    Handler,
    collectiveDeposit,
    collectiveWithdraw,
    targetCollectiveHas,
    updateActor,
    updateTarget,
    updateTargetCollective,
)
from mettagrid.config.mettagrid_config import (
    GridObjectConfig,
    InventoryConfig,
)

if TYPE_CHECKING:
    from cogames.cogs_vs_clips.team import TeamConfig


class CvCHubConfig(CvCStationConfig):
    """Hub station that provides AOE heal/attack, accepts deposits and manages hearts."""

    elements: list[str] = Field(default_factory=lambda: CvCConfig.ELEMENTS)
    heart_cost: dict[str, int] = Field(default_factory=lambda: CvCConfig.HEART_COST)
    attack_deltas: dict[str, int] = Field(default_factory=lambda: CvCConfig.ATTACK_DELTAS.copy())

    def station_cfg(
        self, team: TeamConfig, inventory: Optional[InventoryConfig] = None, map_name: Optional[str] = None
    ) -> GridObjectConfig:
        return GridObjectConfig(
            name="hub",
            map_name=map_name or "hub",
            render_name="hub",
            render_symbol="📦",
            tags=[team.team_tag()],
            collective=team.name,
            inventory=inventory or InventoryConfig(initial={}),
            aoes={
                "territory": AOEConfig(
                    radius=CvCConfig.JUNCTION_DISTANCE,
                ),
                "influence": AOEConfig(
                    radius=CvCConfig.JUNCTION_DISTANCE,
                    filters=[sharedTagPrefix("team:")],
                    mutations=[updateTarget(team.base_aoe_deltas)],
                ),
                "attack": AOEConfig(
                    radius=CvCConfig.JUNCTION_DISTANCE,
                    filters=_opposing_team_filters(),
                    mutations=[updateTarget(self.attack_deltas)],
                ),
            },
            on_use_handlers={
                "deposit": Handler(
                    filters=[sharedTagPrefix("team:"), actorHasAnyOf(self.elements)],
                    mutations=[collectiveDeposit({resource: 100 for resource in self.elements})],
                ),
                "get_heart": Handler(
                    filters=[sharedTagPrefix("team:"), targetCollectiveHas({"heart": 2})],
                    mutations=[collectiveWithdraw({"heart": 1})],
                ),
                "get_and_make_heart": Handler(
                    filters=[
                        sharedTagPrefix("team:"),
                        targetCollectiveHas({"heart": 1}),
                        targetCollectiveHas(self.heart_cost),
                    ],
                    mutations=[updateTargetCollective(_neg(self.heart_cost)), updateActor({"heart": 1})],
                ),
                "get_last_heart": Handler(
                    filters=[sharedTagPrefix("team:"), targetCollectiveHas({"heart": 1})],
                    mutations=[collectiveWithdraw({"heart": 1})],
                ),
                "make_heart": Handler(
                    filters=[sharedTagPrefix("team:"), targetCollectiveHas(self.heart_cost)],
                    mutations=[updateTargetCollective(_neg(self.heart_cost)), updateTargetCollective({"heart": 1})],
                ),
            },
        )
