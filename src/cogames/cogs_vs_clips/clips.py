"""Clips behavior events for CogsGuard missions.

Clips are a non-player faction that gradually takes over neutral junctions.
These events create the spreading/scrambling behavior that pressures players.
"""

from typing import Optional

from pydantic import Field

from mettagrid.base_config import Config
from mettagrid.config.event_config import EventConfig, once, periodic
from mettagrid.config.filter import hasTag, isNear
from mettagrid.config.filter.alignment_filter import isNeutral, isNotAlignedTo, isNotNeutral
from mettagrid.config.mettagrid_config import CollectiveConfig
from mettagrid.config.mutation import alignTo, removeAlignment
from mettagrid.config.query import query
from mettagrid.config.tag import typeTag


class ClipsConfig(Config):
    """Configuration for clips behavior in CogsGuard game mode."""

    disabled: bool = Field(default=False)

    # Clips Behavior - scramble cogs junctions to neutral
    initial_clips_start: int = Field(default=10)
    initial_clips_spots: int = Field(default=1)

    scramble_start: int = Field(default=50)
    scramble_interval: int = Field(default=100)
    scramble_radius: int = Field(default=25)
    scramble_end: Optional[int] = Field(default=None)

    # Clips Behavior - align neutral junctions to clips
    align_start: int = Field(default=100)
    align_interval: int = Field(default=100)
    align_radius: int = Field(default=25)
    align_end: Optional[int] = Field(default=None)

    # Clips Behavior - presence check for re-invasion
    presence_end: Optional[int] = Field(default=None)

    def events(self, cog_teams: list[str], max_steps: int) -> dict[str, EventConfig]:
        """Create all clips events for a mission.

        Returns:
            Dictionary of event name to EventConfig.
        """
        if self.disabled:
            return {}
        scramble_end = max_steps if self.scramble_end is None else min(self.scramble_end, max_steps)
        align_end = max_steps if self.align_end is None else min(self.align_end, max_steps)
        presence_end = max_steps if self.presence_end is None else min(self.presence_end, max_steps)

        clips_junction = query(typeTag("junction"), [hasTag("collective:clips")])
        return {
            "initial_clips": EventConfig(
                name="initial_clips",
                target_query=query(typeTag("junction")),
                timesteps=once(self.initial_clips_start),
                mutations=[alignTo("clips")],
                max_targets=self.initial_clips_spots,
            ),
            "cogs_to_neutral": EventConfig(
                name="cogs_to_neutral",
                target_query=query(typeTag("junction")),
                timesteps=periodic(start=self.scramble_start, period=self.scramble_interval, end=scramble_end),
                # near a clips-aligned junction
                filters=[
                    isNear(clips_junction, radius=self.scramble_radius),
                    isNotAlignedTo("clips"),
                    isNotNeutral(),
                ],
                mutations=[removeAlignment()],
                max_targets=1,
            ),
            "neutral_to_clips": EventConfig(
                name="neutral_to_clips",
                target_query=query(typeTag("junction")),
                timesteps=periodic(start=self.align_start, period=self.align_interval, end=align_end),
                # neutral junctions near a clips-aligned junction
                filters=[
                    # near a clip junction
                    isNear(clips_junction, radius=self.align_radius),
                    # # not near any cog junction
                    # isNot(
                    #     isNear(
                    #         query("type:junction",
                    #             [anyOf([isAlignedTo(cog_team) for cog_team in cog_teams])]),
                    #         radius=CvCConfig.JUNCTION_AOE_RANGE,
                    #     )
                    # ),
                    isNeutral(),
                ],
                mutations=[alignTo("clips")],
                max_targets=1,
            ),
            # If there are no clips-aligned junctions, re-invade
            "presence_check": EventConfig(
                name="presence_check",
                target_query=query(typeTag("junction")),
                timesteps=periodic(start=self.initial_clips_start, period=self.scramble_interval * 2, end=presence_end),
                filters=[isNear(clips_junction, radius=1000)],
                max_targets=1,
                fallback="initial_clips",
            ),
        }

    def collectives(self) -> dict[str, CollectiveConfig]:
        """Create collectives for clips."""
        if self.disabled:
            return {}
        return {"clips": CollectiveConfig(name="clips")}
