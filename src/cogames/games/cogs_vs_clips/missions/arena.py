"""Arena mission definitions for CvC."""

from __future__ import annotations

from cogames.games.cogs_vs_clips.missions.mission import CvCMission
from cogames.games.cogs_vs_clips.missions.terrain import MachinaArena
from mettagrid.mapgen.mapgen import MapGen, MapGenConfig
from mettagrid.mapgen.scenes.building_distributions import DistributionConfig, DistributionType
from mettagrid.mapgen.scenes.compound import CompoundConfig


def _cvc_hub_config() -> CompoundConfig:
    return CompoundConfig(
        hub_object="empty",
        corner_bundle="none",
        cross_bundle="none",
        cross_distance=7,
    )


ARENA_MAP_BUILDER = MapGen.Config(
    width=50,
    height=50,
    instance=MachinaArena.Config(
        spawn_count=20,
        building_coverage=0.1,
        map_corner_offset=1,
        building_distributions={
            "junction": DistributionConfig(type=DistributionType.POISSON),
        },
        hub=_cvc_hub_config(),
    ),
)


def make_arena_map_builder(num_agents: int = 10) -> MapGenConfig:
    """Create a CvC arena map builder with configurable agent count."""
    return MapGen.Config(
        width=50,
        height=50,
        instance=MachinaArena.Config(
            spawn_count=num_agents,
            building_coverage=0.1,
            map_corner_offset=1,
            building_distributions={
                "junction": DistributionConfig(type=DistributionType.POISSON),
            },
            hub=_cvc_hub_config(),
        ),
    )


def make_basic_mission(num_cogs: int = 8, max_steps: int = 1000) -> CvCMission:
    """Arena with full machina1 config (clips, day/night, damage, gear)."""
    return CvCMission(
        name="arena",
        description="CvC Arena - compact training map with gear abilities.",
        map_builder=ARENA_MAP_BUILDER,
        num_cogs=num_cogs,
        min_cogs=1,
        max_cogs=20,
        max_steps=max_steps,
    )
