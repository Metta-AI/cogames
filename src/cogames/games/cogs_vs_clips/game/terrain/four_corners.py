"""Four-corners map builder: 120x120 map with 1-4 corner compounds."""

from __future__ import annotations

from cogames.games.cogs_vs_clips.missions.terrain import MachinaArenaConfig
from mettagrid.mapgen.mapgen import MapGen, MapGenConfig
from mettagrid.mapgen.scene import ChildrenAction
from mettagrid.mapgen.scenes.building_distributions import DistributionConfig, DistributionType
from mettagrid.mapgen.scenes.compound import CompoundConfig
from mettagrid.mapgen.scenes.four_corner_compounds import FourCornerCompoundsConfig


def _empty_hub() -> CompoundConfig:
    """A no-op hub config that renders nothing meaningful."""
    return CompoundConfig(
        hub_object="empty",
        corner_bundle="none",
        cross_bundle="none",
        cross_distance=7,
        include_inner_wall=False,
        spawn_count=0,
        hub_width=7,
        hub_height=7,
    )


def _cvc_compound_template() -> CompoundConfig:
    """Base compound config matching machina1 style."""
    return CompoundConfig(
        hub_object="empty",
        corner_bundle="none",
        cross_bundle="none",
        cross_distance=7,
    )


def build_four_corners_map(num_teams: int, spawn_count_per_team: int) -> MapGenConfig:
    """Build a 120x120 map with corner compounds instead of a center hub."""
    arena = MachinaArenaConfig(
        spawn_count=0,
        map_corner_offset=1,
        hub=_empty_hub(),
        building_coverage=0.05,
        building_distributions={
            "junction": DistributionConfig(type=DistributionType.POISSON),
        },
    )
    four_corners = FourCornerCompoundsConfig(
        compound=_cvc_compound_template(),
        num_compounds=num_teams,
        spawn_count=spawn_count_per_team,
    )
    arena.children.append(ChildrenAction(scene=four_corners, where="full"))
    return MapGen.Config(width=120, height=120, instance=arena, set_team_by_instance=True)


def find_four_corners(arena: MachinaArenaConfig) -> FourCornerCompoundsConfig | None:
    """Find the FourCornerCompoundsConfig in the arena's children."""
    for child in arena.children:
        if isinstance(child.scene, FourCornerCompoundsConfig):
            return child.scene
    return None
