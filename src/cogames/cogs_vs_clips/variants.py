from __future__ import annotations

from typing import TYPE_CHECKING, override

from pydantic import Field

from cogames.cogs_vs_clips.config import CvCConfig
from cogames.cogs_vs_clips.evals.difficulty_variants import DIFFICULTY_VARIANTS
from cogames.cogs_vs_clips.terrain import (
    BaseHubVariant,
    MachinaArenaConfig,
    MachinaArenaVariant,
)
from cogames.core import CoGameMissionVariant
from mettagrid.config.filter import actorHasAnyOf
from mettagrid.config.game_value import inv
from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.map_builder.map_builder import MapBuilderConfig
from mettagrid.mapgen.mapgen import MapGen
from mettagrid.mapgen.scenes.base_hub import DEFAULT_EXTRACTORS as HUB_EXTRACTORS
from mettagrid.mapgen.scenes.building_distributions import DistributionConfig, DistributionType

if TYPE_CHECKING:
    from cogames.cogs_vs_clips.mission import CvCMission


def _set_spawn_count(builder: MapBuilderConfig, spawn_count: int) -> None:
    """Set spawn_count in nested MachinaArenaConfig so map cells match agent count."""
    if isinstance(builder, MapGen.Config) and builder.instance is not None:
        inst = builder.instance
        if isinstance(inst, MachinaArenaConfig):
            builder.instance = inst.model_copy(update={"spawn_count": spawn_count})
        elif isinstance(inst, MapGen.Config) and inst.instance is not None:
            _set_spawn_count(inst, spawn_count)


def _apply_clips_settings(
    mission: CvCMission,
    *,
    initial_clips_start: int | None = None,
    initial_clips_spots: int | None = None,
    scramble_start: int | None = None,
    scramble_interval: int | None = None,
    scramble_radius: int | None = None,
    align_start: int | None = None,
    align_interval: int | None = None,
) -> None:
    clips = mission.clips
    if initial_clips_start is not None:
        clips.initial_clips_start = initial_clips_start
    if initial_clips_spots is not None:
        clips.initial_clips_spots = initial_clips_spots
    if scramble_start is not None:
        clips.scramble_start = scramble_start
    if scramble_interval is not None:
        clips.scramble_interval = scramble_interval
    if scramble_radius is not None:
        clips.scramble_radius = scramble_radius
    if align_start is not None:
        clips.align_start = align_start
    if align_interval is not None:
        clips.align_interval = align_interval


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


class ClipsEasyVariant(CoGameMissionVariant):
    name: str = "clips_easy"
    description: str = "Slow clips expansion with late pressure."

    @override
    def modify_mission(self, mission: CvCMission) -> None:
        _apply_clips_settings(
            mission,
            initial_clips_start=50,
            initial_clips_spots=1,
            scramble_start=250,
            scramble_interval=250,
            scramble_radius=15,
            align_start=300,
            align_interval=250,
        )


class ClipsMediumVariant(CoGameMissionVariant):
    name: str = "clips_medium"
    description: str = "Baseline clips pressure (Machina1 default)."

    @override
    def modify_mission(self, mission: CvCMission) -> None:
        _apply_clips_settings(
            mission,
            initial_clips_start=10,
            initial_clips_spots=1,
            scramble_start=50,
            scramble_interval=100,
            scramble_radius=25,
            align_start=100,
            align_interval=100,
        )


class ClipsHardVariant(CoGameMissionVariant):
    name: str = "clips_hard"
    description: str = "Fast clips pressure with wider territory."

    @override
    def modify_mission(self, mission: CvCMission) -> None:
        _apply_clips_settings(
            mission,
            initial_clips_start=5,
            initial_clips_spots=2,
            scramble_start=25,
            scramble_interval=50,
            scramble_radius=35,
            align_start=50,
            align_interval=50,
        )


class ClipsWaveOnlyVariant(CoGameMissionVariant):
    name: str = "clips_wave_only"
    description: str = "Initial clips wave only, no further spread."

    @override
    def modify_mission(self, mission: CvCMission) -> None:
        disable_start = mission.max_steps + 1
        _apply_clips_settings(
            mission,
            initial_clips_start=10,
            initial_clips_spots=3,
            scramble_start=disable_start,
            scramble_interval=disable_start,
            align_start=disable_start,
            align_interval=disable_start,
            scramble_radius=25,
        )


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


class NoWeatherVariant(CoGameMissionVariant):
    name: str = "no_weather"
    description: str = "Disable the day/night weather cycle."

    @override
    def modify_mission(self, mission: CvCMission) -> None:
        mission.weather.day_deltas = {}
        mission.weather.night_deltas = {}


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


class RandomizeSpawnsVariant(BaseHubVariant):
    name: str = "randomize_spawns"
    description: str = "Randomize agent spawn positions within the hub instead of fixed cardinal directions."

    @override
    def modify_node(self, node):
        node.randomize_spawn_positions = True


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
        agents_per_team = mission.num_agents // self.num_teams
        # Shrink inner instance borders so teams are close together
        if isinstance(original_builder, MapGen.Config):
            original_builder = original_builder.model_copy(deep=True)
            original_builder.border_width = 1
            # Set spawn_count to match agents per team so map cells match agent count
            _set_spawn_count(original_builder, agents_per_team)
        env.game.map_builder = MapGen.Config(
            instance=original_builder,
            instances=self.num_teams,
            set_team_by_instance=True,
            instance_names=[t.short_name for t in mission.teams.values()],
            instance_object_remap={
                "c:hub": "{instance_name}:hub",
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


class NoVibesVariant(CoGameMissionVariant):
    name: str = "no_vibes"
    description: str = "Disable change_vibe action."

    @override
    def modify_env(self, mission: CvCMission, env: MettaGridConfig) -> None:
        env.game.actions.change_vibe.enabled = False


class ForcedRoleVibesVariant(CoGameMissionVariant):
    name: str = "forced_role_vibes"
    description: str = "Assign per-agent roles as a global observation and force each agent's initial vibe by role."

    role_order: list[str] = Field(default_factory=lambda: ["miner", "aligner", "scrambler", "scout"])
    role_id_item: str = Field(default="role_id", description="Inventory item used for the global role_id token.")
    disable_change_vibe: bool = Field(default=True, description="Disable change_vibe so role vibes are forced.")
    per_team: bool = Field(default=True, description="Assign roles by index-within-team.")

    @override
    def modify_env(self, mission: CvCMission, env: MettaGridConfig) -> None:
        allowed_roles = {"miner", "aligner", "scrambler", "scout"}
        if not self.role_order:
            raise ValueError("role_order must be non-empty")
        unknown = [r for r in self.role_order if r not in allowed_roles]
        if unknown:
            raise ValueError(f"Unknown role(s) in role_order: {unknown}. Allowed: {sorted(allowed_roles)}")

        # Make role_id available as a resource, then add it as a per-agent global observation token.
        if self.role_id_item not in env.game.resource_names:
            env.game.resource_names = [*env.game.resource_names, self.role_id_item]

        obs_key = f"inv:own:{self.role_id_item}"
        if obs_key not in env.game.obs.global_obs.obs:
            env.game.obs.global_obs.obs[obs_key] = inv(f"agent.{self.role_id_item}")

        vibe_id_by_name = {name: idx for idx, name in enumerate(env.game.vibe_names)}
        missing_vibes = [r for r in set(self.role_order) if r not in vibe_id_by_name]
        if missing_vibes:
            raise ValueError(
                f"Missing role vibe(s) in env.game.vibe_names: {missing_vibes}. "
                "Expected role names to be present as vibe names."
            )

        # Assign roles and force initial vibe.
        counters: dict[int, int] = {}
        for agent in env.game.agents:
            if self.per_team:
                group_key: int = agent.team_id
            else:
                group_key = 0
            idx = counters.get(group_key, 0)
            counters[group_key] = idx + 1

            role_id = idx % len(self.role_order)
            role_name = self.role_order[role_id]

            agent.vibe = vibe_id_by_name[role_name]
            agent.inventory.initial = {**agent.inventory.initial, self.role_id_item: role_id}

        if self.disable_change_vibe:
            env.game.actions.change_vibe.enabled = False


class ThickSkinnedVariant(CoGameMissionVariant):
    name: str = "thick_skinned"
    description: str = "No passive HP drain. Agents only lose HP in enemy territory."

    @override
    def modify_mission(self, mission: CvCMission) -> None:
        mission.cog.hp_regen = 0


class BraveheartVariant(CoGameMissionVariant):
    name: str = "braveheart"
    description: str = "Hub starts with 255 hearts — enough to align every junction on the map."

    @override
    def modify_mission(self, mission: CvCMission) -> None:
        for team in mission.teams.values():
            team.initial_hearts = 255


_HEART_WITHDRAW_HANDLERS = ("get_heart", "get_and_make_heart", "get_last_heart")


class TinManVariant(CoGameMissionVariant):
    name: str = "tin_man"
    description: str = "No heart without gear — agents must hold a gear to withdraw hearts from the hub."

    @override
    def modify_env(self, mission: CvCMission, env: MettaGridConfig) -> None:
        gear_filter = actorHasAnyOf(CvCConfig.GEAR)
        for obj in env.game.objects.values():
            for handler_name in _HEART_WITHDRAW_HANDLERS:
                handler = obj.on_use_handlers.get(handler_name)
                if handler is not None:
                    handler.filters = [handler.filters[0], gear_filter, *handler.filters[1:]]


VARIANTS: list[CoGameMissionVariant] = [
    BraveheartVariant(),
    CavesVariant(),
    CityVariant(),
    DarkSideVariant(),
    NoWeatherVariant(),
    NoClipsVariant(),
    NoVibesVariant(),
    ThickSkinnedVariant(),
    DesertVariant(),
    EmptyBaseVariant(),
    RandomizeSpawnsVariant(),
    EnergizedVariant(),
    ForestVariant(),
    MultiTeamVariant(),
    QuadrantBuildingsVariant(),
    SingleResourceUniformVariant(),
    Small50Variant(),
    SuperChargedVariant(),
    ForcedRoleVibesVariant(),
    TinManVariant(),
    *DIFFICULTY_VARIANTS,
]

HIDDEN_VARIANTS: list[CoGameMissionVariant] = [
    ClipsEasyVariant(),
    ClipsMediumVariant(),
    ClipsHardVariant(),
    ClipsWaveOnlyVariant(),
]
