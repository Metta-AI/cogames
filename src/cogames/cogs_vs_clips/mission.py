from __future__ import annotations

from pydantic import Field

from cogames.cogs_vs_clips.clips import ClipsConfig
from cogames.cogs_vs_clips.cog import CogConfig
from cogames.cogs_vs_clips.config import CvCConfig
from cogames.cogs_vs_clips.stations import (
    CvCChestConfig,
    CvCExtractorConfig,
    CvCGearStationConfig,
    CvCHubConfig,
    CvCJunctionConfig,
    CvCWallConfig,
)
from cogames.cogs_vs_clips.team import CogTeam
from cogames.cogs_vs_clips.variants import NumCogsVariant
from cogames.core import (
    MAP_MISSION_DELIMITER,
    CoGameMission,
    CoGameMissionVariant,
    CoGameSite,
)
from mettagrid.config.action_config import (
    ActionsConfig,
    ChangeVibeActionConfig,
    MoveActionConfig,
    NoopActionConfig,
)
from mettagrid.config.game_value import inv
from mettagrid.config.mettagrid_config import GameConfig, MettaGridConfig
from mettagrid.config.obs_config import GlobalObsConfig, ObsConfig
from mettagrid.map_builder.map_builder import AnyMapBuilderConfig

__all__ = [
    "MAP_MISSION_DELIMITER",
    "CoGameMission",
    "CoGameMissionVariant",
    "CoGameSite",
    "CvCMission",
    "NumCogsVariant",
]


class CvCMission(CoGameMission):
    """Mission configuration for CogsGuard game mode."""

    max_steps: int = Field(default=10000)

    cog: CogConfig = Field(default_factory=lambda: CogConfig())
    teams: dict[str, CogTeam] = Field(
        default_factory=lambda: {
            "cogs": CogTeam(name="cogs", num_agents=8, wealth=1),
        }
    )

    clips: ClipsConfig = Field(default_factory=lambda: ClipsConfig())

    @property
    def num_agents(self) -> int:
        if self.num_cogs is not None:
            return self.num_cogs
        return sum(team.num_agents for team in self.teams.values())

    def map_builder(self) -> AnyMapBuilderConfig:
        """Return the map builder config. Override in subclasses for custom map generation."""
        return self.site.map_builder

    def make_env(self) -> MettaGridConfig:
        """Create a MettaGridConfig from this mission.

        Applies all variants to the produced configuration.

        Returns:
            MettaGridConfig ready for environment creation
        """

        game = GameConfig(
            map_builder=self.map_builder(),
            max_steps=self.max_steps,
            num_agents=self.num_agents,
            resource_names=CvCConfig.RESOURCES,
            vibe_names=CvCConfig.VIBE_NAMES,
            obs=ObsConfig(
                global_obs=GlobalObsConfig(
                    obs=[inv(f"collective.{resource}") for resource in CvCConfig.ELEMENTS], local_position=True
                )
            ),
            actions=ActionsConfig(
                move=MoveActionConfig(consumed_resources=self.cog.action_cost),
                noop=NoopActionConfig(),
                change_vibe=ChangeVibeActionConfig(vibes=CvCConfig.VIBES),
            ),
            agent=self.cog.agent_config(team="cogs", max_steps=self.max_steps),
            agents=[self.cog.agent_config(team="cogs", max_steps=self.max_steps) for _ in range(self.num_agents)],
            objects={
                "wall": CvCWallConfig().station_cfg(),
                "hub": CvCHubConfig().station_cfg(team="cogs"),
                "junction": CvCJunctionConfig().station_cfg(),
                "chest": CvCChestConfig().station_cfg(team="cogs"),
                **{
                    f"{resource}_extractor": CvCExtractorConfig(resource=resource).station_cfg()
                    for resource in CvCConfig.ELEMENTS
                },
                **{f"{g}_station": CvCGearStationConfig(gear_type=g).station_cfg(team="cogs") for g in CvCConfig.GEAR},
            },
            collectives={
                **{team.name: team.collective_config() for team in self.teams.values()},
                "clips": self.clips.collective_config(),
            },
            events=self.clips.events(max_steps=self.max_steps),
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
