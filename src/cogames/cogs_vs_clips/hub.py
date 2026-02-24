from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from pydantic import Field

from cogames.cogs_vs_clips.config import CvCConfig
from cogames.cogs_vs_clips.stations import CvCStationConfig
from mettagrid.config.filter import actorHasAnyOf, sharedTagPrefix
from mettagrid.config.handler_config import (
    Handler,
    queryDelta,
    queryDeposit,
    queryWithdraw,
    updateActor,
)
from mettagrid.config.mettagrid_config import (
    GridObjectConfig,
    InventoryConfig,
)
from mettagrid.config.territory_config import TerritoryControlConfig

if TYPE_CHECKING:
    from cogames.cogs_vs_clips.team import TeamConfig


def _neg(recipe: dict[str, int]) -> dict[str, int]:
    return {k: -v for k, v in recipe.items()}


class CvCHubConfig(CvCStationConfig):
    """Hub station that provides AOE heal/attack, accepts deposits and manages hearts."""

    elements: list[str] = Field(default_factory=lambda: CvCConfig.ELEMENTS)
    heart_cost: dict[str, int] = Field(default_factory=lambda: CvCConfig.HEART_COST)
    control_range: int = Field(default=CvCConfig.TERRITORY_CONTROL_RADIUS, description="Range for territory control")

    def station_cfg(
        self, team: TeamConfig, inventory: Optional[InventoryConfig] = None, map_name: Optional[str] = None
    ) -> GridObjectConfig:
        hq = team.hub_query()
        return GridObjectConfig(
            name="hub",
            map_name=map_name or "hub",
            render_symbol="📦",
            tags=[
                team.team_tag(),
            ],
            inventory=inventory or InventoryConfig(initial={}),
            territory_controls=[
                TerritoryControlConfig(territory="team_territory", strength=self.control_range * 2),
            ],
            on_use_handlers={
                "deposit": Handler(
                    filters=[sharedTagPrefix("team:"), actorHasAnyOf(self.elements)],
                    mutations=[queryDeposit(hq, {resource: 100 for resource in self.elements})],
                ),
                "get_heart": Handler(
                    filters=[sharedTagPrefix("team:"), *team.hub_has({"heart": 2})],
                    mutations=[queryWithdraw(hq, {"heart": 1})],
                ),
                "get_and_make_heart": Handler(
                    filters=[
                        sharedTagPrefix("team:"),
                        *team.hub_has({"heart": 1}),
                        *team.hub_has(self.heart_cost),
                    ],
                    mutations=[queryDelta(hq, _neg(self.heart_cost)), updateActor({"heart": 1})],
                ),
                "get_last_heart": Handler(
                    filters=[sharedTagPrefix("team:"), *team.hub_has({"heart": 1})],
                    mutations=[queryWithdraw(hq, {"heart": 1})],
                ),
                "make_heart": Handler(
                    filters=[sharedTagPrefix("team:"), *team.hub_has(self.heart_cost)],
                    mutations=[queryDelta(hq, _neg(self.heart_cost)), queryDelta(hq, {"heart": 1})],
                ),
            },
        )
