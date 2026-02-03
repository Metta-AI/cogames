from __future__ import annotations

from typing import TYPE_CHECKING, override

from pydantic import Field

from cogames.cogs_vs_clips.config import CvCConfig
from cogames.cogs_vs_clips.evals.difficulty_variants import DIFFICULTY_VARIANTS
from cogames.cogs_vs_clips.terrain import BaseHubVariant, MachinaArenaVariant
from cogames.core import CoGameMissionVariant
from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.map_builder.map_builder import MapBuilderConfig
from mettagrid.mapgen.mapgen import MapGen
from mettagrid.mapgen.scenes.base_hub import DEFAULT_EXTRACTORS as HUB_EXTRACTORS
from mettagrid.mapgen.scenes.building_distributions import DistributionConfig, DistributionType

if TYPE_CHECKING:
    from cogames.cogs_vs_clips.mission import CvCMission


class NumCogsVariant(CoGameMissionVariant):
    name: str = "num_cogs"
    description: str = "Set the number of cogs for the mission."
    num_cogs: int

    @override
    def modify_mission(self, mission: CvCMission) -> None:
        if self.num_cogs < mission.site.min_cogs or self.num_cogs > mission.site.max_cogs:
            raise ValueError(
                f"Invalid number of cogs for {mission.site.name}: {self.num_cogs}. "
                + f"Must be between {mission.site.min_cogs} and {mission.site.max_cogs}"
            )

        mission.num_cogs = self.num_cogs


class DarkSideVariant(CoGameMissionVariant):
    name: str = "dark_side"
    description: str = "You're on the dark side of the asteroid. You recharge slower."

    @override
    def modify_mission(self, mission: CvCMission) -> None:
        mission.weather.day_deltas = {"solar": 0}
        mission.weather.night_deltas = {"solar": 0}


class SuperChargedVariant(CoGameMissionVariant):
    name: str = "super_charged"
    description: str = "The sun is shining on you. You recharge faster."

    @override
    def modify_mission(self, mission: CvCMission) -> None:
        mission.weather.day_deltas = {k: v + 2 for k, v in mission.weather.day_deltas.items()}
        mission.weather.night_deltas = {k: v + 2 for k, v in mission.weather.night_deltas.items()}


class EnergizedVariant(CoGameMissionVariant):
    name: str = "energized"
    description: str = "Max energy and full regen so agents never run dry."

    @override
    def modify_mission(self, mission: CvCMission) -> None:
        mission.cog.energy_limit = max(mission.cog.energy_limit, 255)
        mission.weather.day_deltas = {"solar": 255}
        mission.weather.night_deltas = {"solar": 255}


class Small50Variant(CoGameMissionVariant):
    name: str = "small_50"
    description: str = "Set map size to 50x50 for quick runs."

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


class SingleResourceUniformVariant(MachinaArenaVariant):
    name: str = "single_resource_uniform"
    description: str = "Place only a single building via uniform distribution across the map."
    building_name: str = "oxygen_extractor"

    @override
    def modify_node(self, node):
        node.building_names = [self.building_name]
        node.building_weights = {self.building_name: 1.0}
        node.building_distributions = None
        node.distribution = DistributionConfig(type=DistributionType.UNIFORM)


class EmptyBaseVariant(BaseHubVariant):
    name: str = "empty_base"
    description: str = "Base hub with extractors removed from the four corners."
    missing: list[str] = list(HUB_EXTRACTORS)

    @override
    def modify_node(self, node):
        missing_set = set(self.missing or [])
        corner_objects = [name if name not in missing_set else "" for name in HUB_EXTRACTORS]
        node.corner_objects = corner_objects
        node.corner_bundle = "custom"


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


class MultiTeamVariant(CoGameMissionVariant):
    """Split the map into multiple team instances, each with their own hub and resources."""

    name: str = "multi_team"
    description: str = "Split map into separate team instances with independent hubs."
    num_teams: int = Field(default=2, ge=2, le=2, description="Number of teams (max 2 supported)")

    @override
    def modify_mission(self, mission: CvCMission) -> None:
        team = next(iter(mission.teams.values()))
        # Each team gets the original agent count; clear num_cogs so total is derived from teams
        original_agents = mission.num_agents
        mission.teams = {
            name: team.model_copy(update={"name": name, "short_name": name, "num_agents": original_agents})
            for name in ["cogs_green", "cogs_blue"][: self.num_teams]
        }
        mission.num_cogs = None

    def modify_env(self, mission: CvCMission, env: MettaGridConfig) -> None:
        original_builder = env.game.map_builder
        # Shrink inner instance borders so teams are close together
        if isinstance(original_builder, MapGen.Config):
            original_builder.border_width = 1
        env.game.map_builder = MapGen.Config(
            instance=original_builder,
            instances=self.num_teams,
            set_team_by_instance=True,
            instance_names=[t.short_name for t in mission.teams.values()],
            instance_object_remap={
                "c:hub": "{instance_name}:hub",
                "c:chest": "{instance_name}:chest",
                **{f"c:{g}": f"{{instance_name}}:{g}" for g in CvCConfig.GEAR},
            },
            # Connect instances: no added borders, clear walls at boundary
            border_width=0,  # No outer border (inner instances have their own)
            instance_border_width=0,  # No border between instances
            instance_border_clear_radius=3,  # Clear walls near instance boundary
        )


class NoClipsVariant(CoGameMissionVariant):
    name: str = "no_clips"
    description: str = "Disable clips behavior entirely."

    @override
    def modify_mission(self, mission: CvCMission) -> None:
        mission.clips.disabled = True


VARIANTS: list[CoGameMissionVariant] = [
    CavesVariant(),
    CityVariant(),
    DarkSideVariant(),
    NoClipsVariant(),
    DesertVariant(),
    EmptyBaseVariant(),
    EnergizedVariant(),
    ForestVariant(),
    MultiTeamVariant(),
    QuadrantBuildingsVariant(),
    SingleResourceUniformVariant(),
    Small50Variant(),
    SuperChargedVariant(),
    *DIFFICULTY_VARIANTS,
]

HIDDEN_VARIANTS: list[CoGameMissionVariant] = []
