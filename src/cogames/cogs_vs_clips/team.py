"""Team configuration for CogsGuard missions.

Teams are identified by tags (e.g., "team:cogs").
Each team's hub holds the shared inventory.
Collectives are kept for shared resource pools.
"""

from typing import Iterator

from pydantic import Field

from cogames.cogs_vs_clips.config import CvCConfig
from cogames.cogs_vs_clips.hub import CvCHubConfig
from mettagrid.base_config import Config
from mettagrid.config.filter import (
    AnyFilter,
    hasTag,
    hasTagPrefix,
    isNear,
    isNot,
    maxDistance,
    sharedTagPrefix,
)
from mettagrid.config.handler_config import updateTarget
from mettagrid.config.mettagrid_config import (
    AOEConfig,
    CollectiveConfig,
    GridObjectConfig,
    InventoryConfig,
)
from mettagrid.config.mutation import addTag, recomputeMaterializedQuery
from mettagrid.config.mutation.alignment_mutation import alignTo
from mettagrid.config.query import ClosureQuery, MaterializedQuery, materializedQuery, query
from mettagrid.config.tag import typeTag


class TeamConfig(Config):
    """Base configuration for any team (cogs or clips).

    Provides the shared tag/network interface: team_tag(), net_tag(), all_tags().
    Subclasses override _gear_stations() and _hub_inventory() to customize.
    """

    name: str = Field(description="Team name used for tags and team identity")
    short_name: str = Field(description="Short prefix used for map object names")
    team_id: int = Field(default=0, description="Numeric id for this team (set when building game config)")
    base_aoe_range: int = Field(default=CvCConfig.JUNCTION_DISTANCE, description="Range for AOE effects")
    base_aoe_deltas: dict[str, int] = Field(default_factory=lambda: {"energy": 100, "hp": 100})

    def team_tag(self) -> str:
        return f"team:{self.name}"

    def net_tag(self) -> str:
        return f"net:{self.name}"

    def all_tags(self) -> list[str]:
        return [
            self.team_tag(),
            "immune",
        ]

    def base_aoe(self) -> dict[str, AOEConfig]:
        return {
            "influence": AOEConfig(
                radius=self.base_aoe_range,
                filters=[sharedTagPrefix("team:")],
                mutations=[updateTarget(self.base_aoe_deltas)],
                controls_territory=True,
            ),
            "attack": AOEConfig(
                radius=self.base_aoe_range,
                filters=[hasTagPrefix("team:"), isNot(sharedTagPrefix("team:"))],
                mutations=[updateTarget({"hp": -1})],
                controls_territory=True,
            ),
        }

    def materialized_queries(self) -> list[MaterializedQuery]:
        return [
            materializedQuery(
                self.net_tag(),
                ClosureQuery(
                    source=query(
                        typeTag("hub"),
                        hasTag(self.team_tag()),
                    ),
                    candidates=query(typeTag("junction"), hasTag(self.team_tag())),
                    edge_filters=[maxDistance(CvCConfig.JUNCTION_DISTANCE)],
                ),
            )
        ]

    def collective_config(self) -> CollectiveConfig:
        return CollectiveConfig(name=self.name)

    def junction_is_alignable(self) -> list[AnyFilter]:
        return [
            isNot(hasTagPrefix("team:")),
            isNear(query(self.net_tag()), radius=CvCConfig.JUNCTION_DISTANCE),
        ]

    def junction_align_mutations(self) -> list:
        return [
            addTag(self.team_tag()),
            addTag(self.net_tag()),
            alignTo(self.name),
            recomputeMaterializedQuery(self.net_tag()),
        ]

    # def inventory_value(self, resource: str) -> QueryInventoryValue:
    #     """GameValue that reads a resource from this team's hub inventory."""
    #     return QueryInventoryValue(query=query(typeTag(f"{self.short_name}:hub")), resource=resource)

    def stations(self) -> dict[str, GridObjectConfig]:
        return {
            **self._hub_station(),
            **dict(self._gear_stations()),
        }

    def _hub_station(self) -> dict[str, GridObjectConfig]:
        map_name = f"{self.short_name}:hub"
        cfg = CvCHubConfig().station_cfg(team=self, inventory=self._hub_inventory(), map_name=map_name)
        return {map_name: cfg}

    def _gear_stations(self) -> Iterator[tuple[str, GridObjectConfig]]:
        return iter([])

    def _hub_inventory(self) -> InventoryConfig:
        return InventoryConfig(initial={})
