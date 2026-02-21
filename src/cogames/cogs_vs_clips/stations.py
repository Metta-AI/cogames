from __future__ import annotations

from pydantic import Field

from cogames.cogs_vs_clips.config import CvCConfig
from mettagrid.base_config import Config
from mettagrid.config.filter import sharedTagPrefix
from mettagrid.config.handler_config import (
    Handler,
    actorHas,
    collectiveWithdraw,
    targetCollectiveHas,
    updateActor,
    updateTargetCollective,
    withdraw,
)
from mettagrid.config.mettagrid_config import (
    GridObjectConfig,
    InventoryConfig,
    WallConfig,
)


def _neg(recipe: dict[str, int]) -> dict[str, int]:
    return {k: -v for k, v in recipe.items()}


class CvCStationConfig(Config):
    def station_cfg(self) -> GridObjectConfig:
        raise NotImplementedError("Subclasses must implement this method")


class CvCWallConfig(CvCStationConfig):
    def station_cfg(self) -> WallConfig:
        return WallConfig(name="wall", render_symbol="⬛")


class CvCExtractorConfig(CvCStationConfig):
    """Simple resource extractor with inventory that transfers resources to actors."""

    resource: str = Field(description="The resource to extract")
    initial_amount: int = Field(default=100, description="Initial amount of resource in extractor")
    small_amount: int = Field(default=1, description="Amount extracted without mining equipment")
    large_amount: int = Field(default=10, description="Amount extracted with mining equipment")

    def station_cfg(self) -> GridObjectConfig:
        return GridObjectConfig(
            name=f"{self.resource}_extractor",
            render_symbol="📦",
            on_use_handlers={
                # Order matters: miner first so agents with miner gear get the bonus
                "miner": Handler(
                    filters=[actorHas({"miner": 1})],
                    mutations=[withdraw({self.resource: self.large_amount}, remove_when_empty=True)],
                ),
                "extract": Handler(
                    filters=[],
                    mutations=[withdraw({self.resource: self.small_amount}, remove_when_empty=True)],
                ),
            },
            inventory=InventoryConfig(initial={self.resource: self.initial_amount}),
        )


class CvCChestConfig(CvCStationConfig):
    """Chest station for heart management.

    Uses collective operations to access the team's shared inventory.
    """

    heart_cost: dict[str, int] = Field(default_factory=lambda: CvCConfig.HEART_COST)

    def station_cfg(self, team: str, team_name: str | None = None) -> GridObjectConfig:
        tag_team = team_name or team
        # hub_query = query(f"type:{team}:hub")
        return GridObjectConfig(
            name=f"{team}:chest",
            render_name="chest",
            render_symbol="📦",
            tags=[f"team:{tag_team}"],
            collective=tag_team,
            on_use_handlers={
                # Using collective-based operations (query-based alternatives commented out)
                "get_heart": Handler(
                    filters=[sharedTagPrefix("team:"), targetCollectiveHas({"heart": 1})],
                    # query-based: filters=[sharedTagPrefix("team:"), queryHas(hub_query, {"heart": 1})],
                    mutations=[collectiveWithdraw({"heart": 1})],
                    # query-based: mutations=[queryWithdraw(hub_query, {"heart": 1})],
                ),
                "make_heart": Handler(
                    filters=[sharedTagPrefix("team:"), targetCollectiveHas(self.heart_cost)],
                    # query-based: filters=[sharedTagPrefix("team:"), queryHas(hub_query, self.heart_cost)],
                    mutations=[
                        updateTargetCollective(_neg(self.heart_cost)),
                        # query-based: queryDelta(hub_query, _neg(self.heart_cost)),
                        updateActor({"heart": 1}),
                    ],
                ),
            },
        )
