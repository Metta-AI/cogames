from __future__ import annotations

from abc import ABC
from typing import TypeVar, override

from pydantic import Field
from typing_extensions import Self

from cogames.cogs_vs_clips.cog import CogConfig
from cogames.cogs_vs_clips.stations import (
    ELEMENTS,
    GEAR,
    CogsGuardChestConfig,
    CvCWallConfig,
    GearStationConfig,
    HubConfig,
    JunctionConfig,
    SimpleExtractorConfig,
)
from mettagrid.base_config import Config
from mettagrid.config.action_config import (
    ActionsConfig,
    ChangeVibeActionConfig,
    MoveActionConfig,
    NoopActionConfig,
)
from mettagrid.config.event_config import EventConfig, periodic
from mettagrid.config.filter import isAlignedTo, isNear
from mettagrid.config.game_value import inv
from mettagrid.config.game_value import stat as game_stat
from mettagrid.config.mettagrid_config import (
    CollectiveConfig,
    GameConfig,
    InventoryConfig,
    MettaGridConfig,
    ResourceLimitsConfig,
)
from mettagrid.config.mutation import alignTo
from mettagrid.config.obs_config import GlobalObsConfig, ObsConfig
from mettagrid.config.reward_config import numObjects, reward
from mettagrid.config.tag import typeTag
from mettagrid.config.vibes import Vibe
from mettagrid.map_builder.map_builder import AnyMapBuilderConfig

# Type variable for mission types
TMission = TypeVar("TMission", bound="MissionBase")


class MissionVariant(Config, ABC):
    # Note: we could derive the name from the class name automatically, but it would make it
    # harder to find the variant source code based on CLI interactions.
    name: str
    description: str = Field(default="")

    def modify_mission(self, mission: MissionBase) -> None:
        # Override this method to modify the mission.
        # Variants are allowed to modify the mission in-place - it's guaranteed to be a one-time only instance.
        pass

    def modify_env(self, mission: MissionBase, env: MettaGridConfig) -> None:
        # Override this method to modify the produced environment.
        # Variants are allowed to modify the environment in-place.
        pass

    def compat(self, mission: MissionBase) -> bool:
        """Check if this variant is compatible with the given mission.

        Returns True if the variant can be safely applied to the mission.
        Override this method to add compatibility checks.
        """
        return True

    def apply(self, mission: TMission) -> TMission:
        mission = mission.model_copy(deep=True)
        mission.variants.append(self)
        self.modify_mission(mission)
        return mission

    # Temporary helper useful as long as we have one-time variants in missions.py file.
    def as_mission(self, name: str, description: str, site: Site) -> Mission:
        return Mission(
            name=name,
            description=description,
            site=site,
            variants=[self],
        )


class NumCogsVariant(MissionVariant):
    name: str = "num_cogs"
    description: str = "Set the number of cogs for the mission."
    num_cogs: int

    @override
    def modify_mission(self, mission: Mission) -> None:
        if self.num_cogs < mission.site.min_cogs or self.num_cogs > mission.site.max_cogs:
            raise ValueError(
                f"Invalid number of cogs for {mission.site.name}: {self.num_cogs}. "
                + f"Must be between {mission.site.min_cogs} and {mission.site.max_cogs}"
            )

        mission.num_cogs = self.num_cogs


class Site(Config):
    name: str
    description: str
    map_builder: AnyMapBuilderConfig

    min_cogs: int = Field(default=1, ge=1)
    max_cogs: int = Field(default=1000, ge=1)


MAP_MISSION_DELIMITER = "."


class MissionBase(Config, ABC):
    """Base class for Mission configurations with common fields and methods."""

    name: str
    description: str
    site: Site
    num_cogs: int | None = None

    # Variants are applied to the mission immediately, and to its env when make_env is called
    variants: list[MissionVariant] = Field(default_factory=list)

    max_steps: int = Field(default=10000)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Can't call `variant.apply` here because it will create a new mission instance
        for variant in self.variants:
            variant.modify_mission(self)

    def with_variants(self, variants: list[MissionVariant]) -> Self:
        mission = self
        for variant in variants:
            mission = variant.apply(mission)
        return mission

    def full_name(self) -> str:
        return f"{self.site.name}{MAP_MISSION_DELIMITER}{self.name}"


# CogsGuard vibes
COGSGUARD_VIBES = [
    Vibe("😐", "default"),
    Vibe("❤️", "heart"),
    Vibe("⚙️", "gear"),
    Vibe("🌀", "scrambler"),
    Vibe("🔗", "aligner"),
    Vibe("⛏️", "miner"),
    Vibe("🔭", "scout"),
]


class Mission(MissionBase):
    """Mission configuration for CogsGuard game mode."""

    # Agent configuration
    cog: CogConfig = Field(default_factory=CogConfig)

    wealth: int = Field(default=1)

    # Collective initial resources
    collective_initial_carbon: int = Field(default=10)
    collective_initial_oxygen: int = Field(default=10)
    collective_initial_germanium: int = Field(default=10)
    collective_initial_silicon: int = Field(default=10)
    collective_initial_heart: int = Field(default=5)

    # Clips Behavior - scramble cogs junctions to neutral
    # Note: must start after initial_clips fires at timestep 10 (events fire alphabetically)
    clips_scramble_start: int = Field(default=50)
    clips_scramble_interval: int = Field(default=100)
    clips_scramble_radius: int = Field(default=25)

    # Clips Behavior - align neutral junctions to clips
    clips_align_start: int = Field(default=100)
    clips_align_interval: int = Field(default=100)
    clips_align_radius: int = Field(default=25)

    # Station configs
    wall: CvCWallConfig = Field(default_factory=CvCWallConfig)

    def make_env(self) -> MettaGridConfig:
        """Create a MettaGridConfig from this mission.

        Applies all variants to the produced configuration.

        Returns:
            MettaGridConfig ready for environment creation
        """
        map_builder = self.site.map_builder
        num_cogs = self.num_cogs if self.num_cogs is not None else self.site.min_cogs

        gear = GEAR
        elements = ELEMENTS
        resources_list = ["energy", "heart", "hp", "influence", *elements, *gear]
        vibe_names = [vibe.name for vibe in COGSGUARD_VIBES]

        extractor_objects = {
            f"{resource}_extractor": SimpleExtractorConfig(resource=resource).station_cfg() for resource in elements
        }
        gear_objects = {f"{g}_station": GearStationConfig(gear_type=g).station_cfg() for g in gear}

        # Create inventory observations for collective resources
        collective_obs = [inv(f"collective.{resource}") for resource in elements]

        game = GameConfig(
            map_builder=map_builder,
            max_steps=self.max_steps,
            num_agents=num_cogs,
            resource_names=resources_list,
            vibe_names=vibe_names,
            obs=ObsConfig(global_obs=GlobalObsConfig(obs=collective_obs, local_position=True)),
            actions=ActionsConfig(
                move=MoveActionConfig(consumed_resources={"energy": self.cog.move_energy_cost}),
                noop=NoopActionConfig(),
                change_vibe=ChangeVibeActionConfig(vibes=COGSGUARD_VIBES),
            ),
            agent=self.cog.agent_config(gear=gear, elements=elements).model_copy(
                update={
                    "rewards": {
                        "aligned_junction_held": reward(
                            game_stat("collective.aligned.junction.held"),
                            weight=1.0 / self.max_steps,
                            denoms=[numObjects("junction")],
                        ),
                    },
                }
            ),
            objects={
                "wall": self.wall.station_cfg(),
                "hub": HubConfig(map_name="hub", team="cogs").station_cfg(),
                "junction": JunctionConfig(map_name="junction").station_cfg(),
                "chest": CogsGuardChestConfig().station_cfg(),
                **extractor_objects,
                **gear_objects,
            },
            collectives={
                "cogs": CollectiveConfig(
                    inventory=InventoryConfig(
                        limits={
                            "resources": ResourceLimitsConfig(min=10000, resources=elements),
                            "hearts": ResourceLimitsConfig(min=65535, resources=["heart"]),
                        },
                        initial={
                            "carbon": self.collective_initial_carbon * self.wealth,
                            "oxygen": self.collective_initial_oxygen * self.wealth,
                            "germanium": self.collective_initial_germanium * self.wealth,
                            "silicon": self.collective_initial_silicon * self.wealth,
                            "heart": self.collective_initial_heart * self.wealth,
                        },
                    ),
                ),
                "clips": CollectiveConfig(),
            },
            events={
                "initial_clips": EventConfig(
                    name="initial_clips",
                    target_tag=typeTag("junction"),
                    timesteps=[10],
                    mutations=[alignTo("clips")],
                    max_targets=1,
                ),
                "cogs_to_neutral": EventConfig(
                    name="cogs_to_neutral",
                    target_tag=typeTag("junction"),
                    timesteps=periodic(
                        start=self.clips_scramble_start,
                        period=self.clips_scramble_interval,
                        end_period=self.clips_scramble_interval // 5,
                        end=self.max_steps,
                    ),
                    filters=[
                        isNear(typeTag("junction"), [isAlignedTo("clips")], radius=self.clips_scramble_radius),
                        isAlignedTo("cogs"),
                    ],
                    mutations=[alignTo(None)],
                    max_targets=1,
                ),
                "neutral_to_clips": EventConfig(
                    name="neutral_to_clips",
                    target_tag=typeTag("junction"),
                    timesteps=periodic(
                        start=self.clips_align_start,
                        period=self.clips_align_interval,
                        end_period=self.clips_align_interval // 5,
                        end=self.max_steps,
                    ),
                    filters=[
                        isNear(typeTag("junction"), [isAlignedTo("clips")], radius=self.clips_align_radius),
                        isAlignedTo(None),
                    ],
                    mutations=[alignTo("clips")],
                    max_targets=1,
                    fallback="cogs_to_neutral",
                ),
                # If the Clips can't find any junctions near them, align a random junction
                "presence_check": EventConfig(
                    name="presence_check",
                    target_tag=typeTag("junction"),
                    timesteps=periodic(
                        start=self.clips_scramble_start,
                        period=self.clips_scramble_interval,
                        end=self.max_steps,
                    ),
                    filters=[
                        isNear(typeTag("junction"), [isAlignedTo("clips")], radius=self.clips_scramble_radius),
                    ],
                    mutations=[],
                    max_targets=1,
                    fallback="initial_clips",
                ),
            },
        )

        env = MettaGridConfig(game=game)
        # Precaution - copy the env, in case the code above uses some global variable that we don't want to modify.
        # This allows variants to modify the env without copying it again.
        env = env.model_copy(deep=True)
        env.label = self.full_name()

        for variant in self.variants:
            variant.modify_env(self, env)
            env.label += f".{variant.name}"

        return env


# Backwards compatibility alias
CogsGuardMission = Mission

AnyMission = Mission
