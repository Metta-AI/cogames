from typing import override

from cogames.cogs_vs_clips.evals.difficulty_variants import DIFFICULTY_VARIANTS
from cogames.cogs_vs_clips.mission import Mission, MissionVariant
from cogames.cogs_vs_clips.procedural import BaseHubVariant, MachinaArenaVariant
from mettagrid.config.action_config import VibeTransfer
from mettagrid.config.game_value import stat
from mettagrid.config.reward_config import reward
from mettagrid.map_builder.map_builder import MapBuilderConfig
from mettagrid.mapgen.mapgen import MapGen
from mettagrid.mapgen.scenes.base_hub import DEFAULT_EXTRACTORS as HUB_EXTRACTORS
from mettagrid.mapgen.scenes.building_distributions import DistributionConfig, DistributionType


class DarkSideVariant(MissionVariant):
    name: str = "dark_side"
    description: str = "You're on the dark side of the asteroid. You recharge slower."

    @override
    def modify_mission(self, mission):
        assert isinstance(mission, Mission)
        mission.cog.energy_regen = 0


class SuperChargedVariant(MissionVariant):
    name: str = "super_charged"
    description: str = "The sun is shining on you. You recharge faster."

    @override
    def modify_mission(self, mission):
        assert isinstance(mission, Mission)
        mission.cog.energy_regen += 2


class RoughTerrainVariant(MissionVariant):
    name: str = "rough_terrain"
    description: str = "The terrain is rough. Moving is more energy intensive."

    @override
    def modify_mission(self, mission):
        assert isinstance(mission, Mission)
        mission.cog.move_energy_cost += 2


class PackRatVariant(MissionVariant):
    name: str = "pack_rat"
    description: str = "Raise heart, cargo, energy, and gear caps to 255."

    @override
    def modify_mission(self, mission):
        assert isinstance(mission, Mission)
        mission.cog.heart_limit = max(mission.cog.heart_limit, 255)
        mission.cog.energy_limit = max(mission.cog.energy_limit, 255)
        mission.cog.cargo_limit = max(mission.cog.cargo_limit, 255)
        mission.cog.gear_limit = max(mission.cog.gear_limit, 255)


class EnergizedVariant(MissionVariant):
    name: str = "energized"
    description: str = "Max energy and full regen so agents never run dry."

    @override
    def modify_mission(self, mission):
        assert isinstance(mission, Mission)
        mission.cog.energy_limit = max(mission.cog.energy_limit, 255)
        mission.cog.energy_regen = mission.cog.energy_limit


class CompassVariant(MissionVariant):
    name: str = "compass"
    description: str = "Enable compass observation."

    @override
    def modify_env(self, mission, env):
        env.game.obs.global_obs.compass = True


class Small50Variant(MissionVariant):
    name: str = "small_50"
    description: str = "Set map size to 50x50 for quick runs."

    def modify_env(self, mission, env) -> None:
        map_builder = env.game.map_builder
        # Only set width/height if instance is a SceneConfig, not a MapBuilderConfig
        # When instance is a MapBuilderConfig, width and height must be None
        if isinstance(map_builder, MapGen.Config) and isinstance(map_builder.instance, MapBuilderConfig):
            # Skip setting width/height for MapBuilderConfig instances
            return
        env.game.map_builder = map_builder.model_copy(update={"width": 50, "height": 50})


# Biome variants (weather) for procedural maps
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
        # Fill almost the entire map with the city layer
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
        # Bias buildings toward the map edges using bimodal clusters centered at
        node.building_coverage = 0.01

        vertical_edges = DistributionConfig(
            type=DistributionType.BIMODAL,
            center1_x=0.92,  # top right corner
            center1_y=0.08,
            center2_x=0.08,  # bottom left corner
            center2_y=0.92,
            cluster_std=0.18,
        )
        horizontal_edges = DistributionConfig(
            type=DistributionType.BIMODAL,
            center1_x=0.08,  # top left corner
            center1_y=0.08,
            center2_x=0.92,  # bottom right corner
            center2_y=0.92,
            cluster_std=0.18,
        )

        # Apply edge-biased distributions to extractors; other buildings follow the global distribution
        names = list(self.building_names)
        node.building_distributions = {
            name: (vertical_edges if i % 2 == 0 else horizontal_edges) for i, name in enumerate(names)
        }
        # Fallback for any unspecified building types
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
            (0.25, 0.25),  # top-left
            (0.75, 0.25),  # top-right
            (0.25, 0.75),  # bottom-left
            (0.75, 0.75),  # bottom-right
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
        # Resolve resource to a concrete building name
        # Restrict building set to only the chosen building and enforce uniform distribution
        node.building_names = [self.building_name]
        node.building_weights = {self.building_name: 1.0}
        node.building_distributions = None
        node.distribution = DistributionConfig(type=DistributionType.UNIFORM)


class EmptyBaseVariant(BaseHubVariant):
    name: str = "empty_base"
    description: str = "Base hub with extractors removed from the four corners."
    # Extractor object names to remove, e.g., ["oxygen_extractor"]
    missing: list[str] = list(HUB_EXTRACTORS)

    @override
    def modify_node(self, node):
        # Use the default extractor order and blank out any that are missing
        missing_set = set(self.missing or [])
        corner_objects = [name if name not in missing_set else "" for name in HUB_EXTRACTORS]
        node.corner_objects = corner_objects
        node.corner_bundle = "custom"


class BalancedCornersVariant(MachinaArenaVariant):
    """Enable corner balancing to ensure fair spawn distances."""

    name: str = "balanced_corners"
    description: str = "Balance path distances from center to corners for fair spawns."
    balance_tolerance: float = 1.5
    max_balance_shortcuts: int = 10

    @override
    def modify_node(self, node):
        node.balance_corners = True
        node.balance_tolerance = self.balance_tolerance
        node.max_balance_shortcuts = self.max_balance_shortcuts


class TraderVariant(MissionVariant):
    name: str = "trader"
    description: str = "Agents can trade resources with each other."

    @override
    def modify_env(self, mission, env):
        # Define vibe transfers for trading resources (actor gives, target receives)
        trade_transfers = [
            VibeTransfer(vibe="carbon_a", target={"carbon": 1}, actor={"carbon": -1}),
            VibeTransfer(vibe="carbon_b", target={"carbon": 10}, actor={"carbon": -10}),
            VibeTransfer(vibe="oxygen_a", target={"oxygen": 1}, actor={"oxygen": -1}),
            VibeTransfer(vibe="oxygen_b", target={"oxygen": 10}, actor={"oxygen": -10}),
            VibeTransfer(vibe="germanium_a", target={"germanium": 1}, actor={"germanium": -1}),
            VibeTransfer(vibe="germanium_b", target={"germanium": 4}, actor={"germanium": -4}),
            VibeTransfer(vibe="silicon_a", target={"silicon": 10}, actor={"silicon": -10}),
            VibeTransfer(vibe="silicon_b", target={"silicon": 50}, actor={"silicon": -50}),
            VibeTransfer(vibe="heart_a", target={"heart": 1}, actor={"heart": -1}),
            VibeTransfer(vibe="heart_b", target={"heart": 4}, actor={"heart": -4}),
        ]
        # Enable transfer action with these vibes
        env.game.actions.transfer.enabled = True
        env.game.actions.transfer.vibe_transfers.extend(trade_transfers)


class SharedRewardsVariant(MissionVariant):
    name: str = "shared_rewards"
    description: str = "Rewards for deposited hearts are shared among all agents."

    @override
    def modify_env(self, mission, env):
        num_cogs = mission.num_cogs if mission.num_cogs is not None else mission.site.min_cogs
        env.game.agent.rewards["chest_heart_deposited_by_agent"] = reward(
            stat("chest.heart.deposited_by_agent"), weight=0
        )
        env.game.agent.rewards["chest_heart_amount"] = reward(stat("chest.heart.amount"), weight=1 / num_cogs)


# TODO - validate that all variant names are unique
VARIANTS: list[MissionVariant] = [
    CavesVariant(),
    CityVariant(),
    CompassVariant(),
    DarkSideVariant(),
    DesertVariant(),
    EmptyBaseVariant(),
    EnergizedVariant(),
    ForestVariant(),
    PackRatVariant(),
    QuadrantBuildingsVariant(),
    RoughTerrainVariant(),
    SharedRewardsVariant(),
    SingleResourceUniformVariant(),
    Small50Variant(),
    SuperChargedVariant(),
    TraderVariant(),
    *DIFFICULTY_VARIANTS,
]

# Hidden variants registry: Remains usable but will NOT appear in `cogames variants` listing
HIDDEN_VARIANTS: list[MissionVariant] = [
    # Example: ExperimentalVariant(),  # keep empty by default
]
