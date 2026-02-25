"""Clips behavior events for CogsGuard missions.

Clips are a non-player faction that gradually takes over neutral junctions.
These events create the spreading/scrambling behavior that pressures players.
"""

from __future__ import annotations

from typing import Iterator, Optional, override

from pydantic import Field

from cogames.cogs_vs_clips.config import CvCConfig
from cogames.cogs_vs_clips.ship import CvCShipConfig
from cogames.cogs_vs_clips.team import TeamConfig
from mettagrid.config.event_config import EventConfig, periodic
from mettagrid.config.filter import AnyFilter, anyOf, hasTag, hasTagPrefix, isNear, isNot, maxDistance
from mettagrid.config.mettagrid_config import GridObjectConfig
from mettagrid.config.mutation import (
    logActorAgentStat,
    recomputeMaterializedQuery,
    removeTagPrefix,
)
from mettagrid.config.query import ClosureQuery, MaterializedQuery, Query, materializedQuery, query
from mettagrid.config.tag import typeTag


class ClipsConfig(TeamConfig):
    """Configuration for clips behavior in CogsGuard game mode."""

    name: str = Field(default="clips")
    short_name: str = Field(default="clips")
    disabled: bool = Field(default=False)

    initial_clips_start: int = Field(default=10)
    initial_clips_spots: int = Field(default=1)

    scramble_start: int = Field(default=50)
    scramble_interval: int = Field(default=100)
    scramble_radius: int = Field(default=CvCConfig.JUNCTION_ALIGN_DISTANCE)
    scramble_end: Optional[int] = Field(default=None)

    align_start: int = Field(default=100)
    align_interval: int = Field(default=100)
    align_end: Optional[int] = Field(default=None)

    presence_end: Optional[int] = Field(default=None)
    align_all_neutral: bool = Field(default=False)
    align_unlimited_targets: bool = Field(default=False)

    @override
    def hub_query(self) -> Query:
        return query(typeTag("ship"), hasTag(self.team_tag()))

    @override
    def materialized_queries(self) -> list[MaterializedQuery]:
        return [
            materializedQuery(
                self.net_tag(),
                ClosureQuery(
                    source=self.hub_query(),
                    candidates=query(typeTag("junction"), hasTag(self.team_tag())),
                    edge_filters=[maxDistance(CvCConfig.JUNCTION_ALIGN_DISTANCE)],
                ),
            )
        ]

    @override
    def junction_is_alignable(self) -> list[AnyFilter]:
        return [
            isNot(hasTagPrefix("team:")),
            anyOf(
                [
                    isNear(query(self.net_tag()), radius=CvCConfig.JUNCTION_ALIGN_DISTANCE),
                    isNear(self.hub_query(), radius=CvCConfig.JUNCTION_ALIGN_DISTANCE),
                ]
            ),
        ]

    def events(self, max_steps: int, ship_count: int = 1) -> dict[str, EventConfig]:
        if self.disabled:
            return {}
        scramble_end = max_steps if self.scramble_end is None else min(self.scramble_end, max_steps)
        align_end = max_steps if self.align_end is None else min(self.align_end, max_steps)
        align_filters: list[AnyFilter] = (
            [isNot(hasTagPrefix("team:"))] if self.align_all_neutral else self.junction_is_alignable()
        )
        targets = max(1, ship_count)
        align_max_targets = None if self.align_unlimited_targets else targets

        return {
            "neutral_to_clips": EventConfig(
                name="neutral_to_clips",
                target_query=query(typeTag("junction"), filters=align_filters),
                timesteps=periodic(start=self.align_start, period=self.align_interval, end=align_end),
                mutations=self.junction_align_mutations(),
                max_targets=align_max_targets,
            ),
            "cogs_to_neutral": EventConfig(
                name="cogs_to_neutral",
                target_query=query(
                    typeTag("junction"),
                    filters=[
                        hasTagPrefix("team:"),
                        isNot(hasTag(self.team_tag())),
                        isNear(query(self.net_tag()), radius=self.scramble_radius),
                    ],
                ),
                timesteps=periodic(start=self.scramble_start, period=self.scramble_interval, end=scramble_end),
                mutations=[
                    removeTagPrefix("net:"),
                    logActorAgentStat("junction.scrambled_by_clips"),
                    recomputeMaterializedQuery("net:"),
                ],
                max_targets=targets,
            ),
        }

    @override
    def _hub_station(self) -> dict[str, GridObjectConfig]:
        map_name = f"{self.short_name}:ship"
        cfg = CvCShipConfig().station_cfg(team=self, map_name=map_name)
        return {map_name: cfg}

    @override
    def _gear_stations(self) -> Iterator[tuple[str, GridObjectConfig]]:
        return iter([])
