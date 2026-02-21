"""Clips behavior events for CogsGuard missions.

Clips are a non-player faction that gradually takes over neutral junctions.
These events create the spreading/scrambling behavior that pressures players.
"""

from __future__ import annotations

from typing import Iterator, Optional, override

from pydantic import Field

from cogames.cogs_vs_clips.config import CvCConfig
from cogames.cogs_vs_clips.team import TeamConfig
from mettagrid.config.event_config import EventConfig, once, periodic
from mettagrid.config.filter import hasTag, hasTagPrefix, isNear, isNot
from mettagrid.config.mettagrid_config import GridObjectConfig
from mettagrid.config.mutation import (
    addTag,
    alignTo,
    logActorAgentStat,
    recomputeMaterializedQuery,
    removeTagPrefix,
)
from mettagrid.config.query import query
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
    scramble_radius: int | None = Field(default=None)
    scramble_end: Optional[int] = Field(default=None)

    align_start: int = Field(default=100)
    align_interval: int = Field(default=100)
    align_end: Optional[int] = Field(default=None)

    presence_end: Optional[int] = Field(default=None)

    def events(self, max_steps: int) -> dict[str, EventConfig]:
        if self.disabled:
            return {}
        scramble_radius = 2 * CvCConfig.JUNCTION_DISTANCE if self.scramble_radius is None else self.scramble_radius
        scramble_end = max_steps if self.scramble_end is None else min(self.scramble_end, max_steps)
        align_end = max_steps if self.align_end is None else min(self.align_end, max_steps)

        return {
            "initial_clips": EventConfig(
                name="initial_clips",
                target_query=query(typeTag("junction"), filters=[isNot(hasTagPrefix("team:"))]),
                timesteps=once(self.initial_clips_start),
                # Seed a scramble target; don't force net connectivity at spawn time.
                mutations=[addTag(self.team_tag()), alignTo(self.name)],
                max_targets=self.initial_clips_spots,
            ),
            "neutral_to_clips": EventConfig(
                name="neutral_to_clips",
                target_query=query(typeTag("junction"), filters=self.junction_is_alignable()),
                timesteps=periodic(start=self.align_start, period=self.align_interval, end=align_end),
                mutations=self.junction_align_mutations(),
                max_targets=1,
            ),
            "cogs_to_neutral": EventConfig(
                name="cogs_to_neutral",
                target_query=query(
                    "type:junction",
                    filters=[
                        isNot(hasTag(self.team_tag())),
                        isNear(query(self.net_tag()), radius=scramble_radius),
                    ],
                ),
                timesteps=periodic(start=self.scramble_start, period=self.scramble_interval, end=scramble_end),
                mutations=[
                    removeTagPrefix("net:"),
                    logActorAgentStat("junction.scrambled_by_agent"),
                    recomputeMaterializedQuery("net:"),
                ],
                max_targets=1,
            ),
        }

    @override
    def _gear_stations(self) -> Iterator[tuple[str, GridObjectConfig]]:
        return iter([])
