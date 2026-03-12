"""Terrain / biome variants and building placement."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from pydantic import Field

from cogames.core import CoGameMissionVariant, Deps
from cogames.games.cogs_vs_clips.missions.terrain import MachinaArenaVariant, find_machina_arena
from cogames.variants import ResolvedDeps
from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.map_builder.map_builder import MapBuilderConfig
from mettagrid.mapgen.mapgen import MapGen
from mettagrid.mapgen.scenes.building_distributions import DistributionConfig, DistributionType

if TYPE_CHECKING:
    from cogames.games.cogs_vs_clips.missions.mission import CvCMission
    from mettagrid.map_builder.map_builder import AnyMapBuilderConfig


class BuildingsVariant(CoGameMissionVariant):
    """Configure which buildings to place on the map and their weights."""

    name: str = "buildings"
    description: str = "Configure building placement on the map."
    building_density: dict[str, float] = Field(
        default_factory=dict, description="Building name -> weight, configured by other variants."
    )

    @override
    def dependencies(self) -> Deps:
        return Deps()

    @override
    def configure(self, deps: ResolvedDeps) -> None:
        pass

    @override
    def modify_env(self, mission: CvCMission, env: MettaGridConfig) -> None:
        if self.building_density:
            _apply_buildings(env.game.map_builder, self.building_density)


def _apply_buildings(builder: AnyMapBuilderConfig, buildings: dict[str, float]) -> None:
    """Apply building weights to a MapGen-based map builder."""
    arena = find_machina_arena(builder)
    if arena is None:
        return
    weights = dict(arena.building_weights or {})
    for name, weight in buildings.items():
        weights.setdefault(name, weight)
    arena.building_weights = weights


class Small50Variant(CoGameMissionVariant):
    name: str = "small_50"
    description: str = "Set map size to 50x50 for quick runs."

    @override
    def modify_env(self, mission, env) -> None:
        map_builder = env.game.map_builder
        if isinstance(map_builder, MapGen.Config) and isinstance(map_builder.instance, MapBuilderConfig):
            return
        env.game.map_builder = map_builder.model_copy(update={"width": 50, "height": 50})


class DesertVariant(MachinaArenaVariant):
    name: str = "desert"
    description: str = "The desert sands make navigation challenging."

    @override
    def modify_node(self, node):
        node.biome_weights = {"desert": 1.0, "caves": 0.0, "forest": 0.0, "city": 0.0}
        node.base_biome = "desert"


class ForestVariant(MachinaArenaVariant):
    name: str = "forest"
    description: str = "Dense forests obscure your view."

    @override
    def modify_node(self, node):
        node.biome_weights = {"forest": 1.0, "caves": 0.0, "desert": 0.0, "city": 0.0}
        node.base_biome = "forest"


class CityVariant(MachinaArenaVariant):
    name: str = "city"
    description: str = "Ancient city ruins provide structured pathways."

    def modify_node(self, node):
        node.biome_weights = {"city": 1.0, "caves": 0.0, "desert": 0.0, "forest": 0.0}
        node.base_biome = "city"
        node.density_scale = 1.0
        node.biome_count = 1
        node.max_biome_zone_fraction = 0.95


class CavesVariant(MachinaArenaVariant):
    name: str = "caves"
    description: str = "Winding cave systems create a natural maze."

    @override
    def modify_node(self, node):
        node.biome_weights = {"caves": 1.0, "desert": 0.0, "forest": 0.0, "city": 0.0}
        node.base_biome = "caves"


class DistantResourcesVariant(MachinaArenaVariant):
    name: str = "distant_resources"
    description: str = "Resources scattered far from base; heavy routing coordination."
    building_names: list[str] = ["carbon_extractor", "oxygen_extractor", "germanium_extractor", "silicon_extractor"]

    @override
    def modify_node(self, node):
        node.building_coverage = 0.01

        vertical_edges = DistributionConfig(
            type=DistributionType.BIMODAL,
            center1_x=0.92,
            center1_y=0.08,
            center2_x=0.08,
            center2_y=0.92,
            cluster_std=0.18,
        )
        horizontal_edges = DistributionConfig(
            type=DistributionType.BIMODAL,
            center1_x=0.08,
            center1_y=0.08,
            center2_x=0.92,
            center2_y=0.92,
            cluster_std=0.18,
        )

        names = list(self.building_names)
        node.building_distributions = {
            name: (vertical_edges if i % 2 == 0 else horizontal_edges) for i, name in enumerate(names)
        }
        node.distribution = DistributionConfig(type=DistributionType.UNIFORM)


class QuadrantBuildingsVariant(MachinaArenaVariant):
    name: str = "quadrant_buildings"
    description: str = "Place buildings in the four quadrants of the map."
    building_names: list[str] = ["carbon_extractor", "oxygen_extractor", "germanium_extractor", "silicon_extractor"]

    @override
    def modify_node(self, node):
        node.building_names = self.building_names

        names = list(node.building_names or self.building_names)
        centers = [
            (0.25, 0.25),
            (0.75, 0.25),
            (0.25, 0.75),
            (0.75, 0.75),
        ]
        dists: dict[str, DistributionConfig] = {}
        for i, name in enumerate(names):
            cx, cy = centers[i % len(centers)]
            dists[name] = DistributionConfig(
                type=DistributionType.NORMAL,
                mean_x=cx,
                mean_y=cy,
                std_x=0.18,
                std_y=0.18,
            )
        node.building_distributions = dists
        node.distribution = DistributionConfig(type=DistributionType.UNIFORM)


# TODO: unchecked variant
class BalancedCornersVariant(MachinaArenaVariant):
    name: str = "balanced_corners"
    description: str = "Balance path distances from center to corners for fair spawns."
    balance_tolerance: float = 1.5
    max_balance_shortcuts: int = 10

    @override
    def modify_node(self, node):
        node.balance_corners = True
        node.balance_tolerance = self.balance_tolerance
        node.max_balance_shortcuts = self.max_balance_shortcuts
