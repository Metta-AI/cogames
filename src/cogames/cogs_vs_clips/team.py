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
    anyOf,
    hasTag,
    hasTagPrefix,
    isNear,
    isNot,
    maxDistance,
)
from mettagrid.config.game_value import MaxGameValue, QueryCountValue, QueryInventoryValue, SumGameValue, stat, val
from mettagrid.config.handler_config import Handler
from mettagrid.config.mettagrid_config import (
    GridObjectConfig,
    InventoryConfig,
)
from mettagrid.config.mutation import StatsMutation, StatsTarget, addTag, logStatToGame, recomputeMaterializedQuery
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
                    edge_filters=[maxDistance(max(CvCConfig.JUNCTION_ALIGN_DISTANCE, CvCConfig.HUB_ALIGN_DISTANCE))],
                ),
            )
        ]

    def on_tick_handlers(self, resource_names: list[str]) -> dict[str, Handler]:
        junction_name = f"{self.name}/aligned.junction"
        junction_value = MaxGameValue(
            values=[
                SumGameValue(values=[QueryCountValue(query=query(self.net_tag())), val(-1)]),
                val(0),
            ]
        )
        return {
            # SET: junction count = max(net_count - 1, 0)
            f"{self.name}/junction_count": Handler(
                mutations=[StatsMutation(stat=junction_name, source=junction_value, target=StatsTarget.GAME)]
            ),
            # ADD: held = held + junction_count
            f"{self.name}/junction_held": Handler(
                mutations=[logStatToGame(f"{self.name}/aligned.junction.held", source=stat(f"game.{junction_name}"))]
            ),
            # SET: resource amounts from hub inventory
            **{
                f"{self.name}/{resource}_amount": Handler(
                    mutations=[
                        StatsMutation(
                            stat=f"{self.name}/{resource}.amount",
                            source=QueryInventoryValue(
                                query=query(typeTag("hub"), hasTag(self.team_tag())),
                                item=resource,
                            ),
                            target=StatsTarget.GAME,
                        )
                    ]
                )
                for resource in resource_names
            },
        }

    def hub_query(self) -> Query:
        return query(typeTag("hub"), hasTag(self.team_tag()))

    def junction_is_alignable(self) -> list[AnyFilter]:
        return [
            isNot(hasTagPrefix("team:")),
            anyOf(
                [
                    isNear(query(self.net_tag()), radius=CvCConfig.JUNCTION_ALIGN_DISTANCE),
                    isNear(query(typeTag("hub"), hasTag(self.team_tag())), radius=CvCConfig.HUB_ALIGN_DISTANCE),
                ]
            ),
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
