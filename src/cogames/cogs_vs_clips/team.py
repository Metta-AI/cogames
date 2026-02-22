"""Team configuration for CogsGuard missions.

Teams are identified by tags (e.g., "team:cogs").
Each team's hub holds the shared inventory.
"""

from typing import Iterator

from pydantic import Field

from cogames.cogs_vs_clips.config import CvCConfig
from cogames.cogs_vs_clips.hub import CvCHubConfig
from mettagrid.base_config import Config
from mettagrid.config.filter import (
    AnyFilter,
    GameValueFilter,
    HandlerTarget,
    hasTag,
    hasTagPrefix,
    isNear,
    isNot,
    maxDistance,
)
from mettagrid.config.game_value import QueryInventoryValue
from mettagrid.config.mettagrid_config import (
    GridObjectConfig,
    InventoryConfig,
)
from mettagrid.config.mutation import addTag, recomputeMaterializedQuery
from mettagrid.config.query import ClosureQuery, MaterializedQuery, Query, materializedQuery, query
from mettagrid.config.tag import typeTag


class TeamConfig(Config):
    """Base configuration for any team (cogs or clips).

    Provides the shared tag/network interface: team_tag(), net_tag(), all_tags().
    Subclasses override _gear_stations() and _hub_inventory() to customize.
    """

    name: str = Field(description="Team name used for tags and team identity")
    short_name: str = Field(description="Short prefix used for map object names")
    team_id: int = Field(default=0, description="Numeric id for this team (set when building game config)")

    def team_tag(self) -> str:
        return f"team:{self.name}"

    def net_tag(self) -> str:
        return f"net:{self.name}"

    def all_tags(self) -> list[str]:
        return [
            self.team_tag(),
            "immune",
        ]

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
                    edge_filters=[maxDistance(CvCConfig.JUNCTION_ALIGN_DISTANCE)],
                ),
            )
        ]

    def hub_query(self) -> Query:
        return query(typeTag("hub"), hasTag(self.team_tag()))

    def junction_is_alignable(self) -> list[AnyFilter]:
        return [
            isNot(hasTagPrefix("team:")),
            isNear(query(self.net_tag()), radius=CvCConfig.JUNCTION_ALIGN_DISTANCE),
        ]

    def junction_align_mutations(self) -> list:
        return [
            addTag(self.team_tag()),
            addTag(self.net_tag()),
            recomputeMaterializedQuery(self.net_tag()),
        ]

    def hub_has(self, resources: dict[str, int]) -> list[GameValueFilter]:
        """Filters: team's hub has at least the given amount of each resource."""
        hq = self.hub_query()
        return [
            GameValueFilter(
                target=HandlerTarget.TARGET,
                value=QueryInventoryValue(query=hq, item=resource),
                min=amount,
            )
            for resource, amount in resources.items()
        ]

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
