"""Empty mission - base game with no variants."""

from __future__ import annotations

from cogames.games.cogs_vs_clips.missions.mission import CvCMission
from cogames.games.cogs_vs_clips.missions.terrain import MachinaArena
from mettagrid.mapgen.mapgen import MapGen

EMPTY_MAP_BUILDER = MapGen.Config(
    width=50,
    height=50,
    instance=MachinaArena.Config(spawn_count=8),
)


def make_empty_mission(num_cogs: int = 8, max_steps: int = 1000) -> CvCMission:
    return CvCMission(
        name="empty",
        description="Base game with no variants.",
        map_builder=EMPTY_MAP_BUILDER,
        default_variant=None,
        num_cogs=num_cogs,
        min_cogs=1,
        max_cogs=8,
        max_steps=max_steps,
    )
