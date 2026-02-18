from __future__ import annotations

from pydantic import Field

from cogames.cogs_vs_clips.clips import ClipsConfig
from cogames.cogs_vs_clips.cog import CogConfig
from cogames.cogs_vs_clips.config import CvCConfig
from cogames.cogs_vs_clips.stations import (
    CvCExtractorConfig,
    CvCGearStationConfig,
    CvCHubConfig,
    CvCJunctionConfig,
    CvCWallConfig,
)
from cogames.cogs_vs_clips.team import CogTeam
from cogames.cogs_vs_clips.variants import NumCogsVariant
from cogames.cogs_vs_clips.weather import WeatherConfig
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
from mettagrid.config.tag import typeTag
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
    total_junctions: int = Field(default=118, description="Total junctions on the map (for curriculum scaling)")

    cog: CogConfig = Field(default_factory=lambda: CogConfig())
    teams: dict[str, CogTeam] = Field(
        default_factory=lambda: {
            "cogs": CogTeam(name="cogs", num_agents=8, wealth=1),
        }
    )

    clips: ClipsConfig = Field(default_factory=lambda: ClipsConfig())
    weather: WeatherConfig = Field(default_factory=lambda: WeatherConfig())

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
        team_objs = list(self.teams.values())
        base_junction_cfg = CvCJunctionConfig().station_cfg(team="cogs")
        base_junction = base_junction_cfg.model_copy(
            update={
                "name": "c:junction",
                "map_name": "c:junction",
                # Keep this junction behaving like a "junction" for type-tag filters (e.g. clips events).
                "tags": base_junction_cfg.tags + [typeTag("junction")],
            }
        )
        game = GameConfig(
            map_builder=self.map_builder(),
            max_steps=self.max_steps,
            num_agents=self.num_agents,
            resource_names=CvCConfig.RESOURCES,
            vibe_names=CvCConfig.VIBE_NAMES,
            obs=ObsConfig(
                global_obs=GlobalObsConfig(
                    obs=[inv(f"collective.{resource}") for resource in CvCConfig.ELEMENTS],
                    local_position=True,
                    last_action_move=True,
                ),
                aoe_mask=True,
                territory_map=True,
            ),
            actions=ActionsConfig(
                move=MoveActionConfig(consumed_resources=self.cog.action_cost),
                noop=NoopActionConfig(),
                change_vibe=ChangeVibeActionConfig(vibes=CvCConfig.VIBES),
            ),
            agents=[
                self.cog.agent_config(team=t.name, max_steps=self.max_steps)
                for t in team_objs
                for _ in range(t.num_agents)
            ],
            objects={
                "wall": CvCWallConfig().station_cfg(),
                "junction": CvCJunctionConfig().station_cfg(),
                "c:junction": base_junction,
                **{
                    f"{resource}_extractor": CvCExtractorConfig(resource=resource).station_cfg()
                    for resource in CvCConfig.ELEMENTS
                },
                **{
                    f"{t.short_name}:hub": CvCHubConfig(
                        initial_hearts=(
                            t.initial_hearts if t.initial_hearts is not None else CvCConfig.INITIAL_HEARTS * t.wealth
                        )
                    ).station_cfg(team=t.short_name, collective=t.name)
                    for t in team_objs
                },
                **{
                    f"{t.short_name}:{g}": CvCGearStationConfig(gear_type=g).station_cfg(
                        team=t.short_name, collective=t.name
                    )
                    for t in team_objs
                    for g in CvCConfig.GEAR
                },
            },
            collectives={
                **{t.name: t.collective_config() for t in team_objs},
                **self.clips.collectives(),
            },
            events=self._merge_events(),
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

    def _merge_events(self) -> dict:
        """Merge clips and weather events, raising on key conflicts."""
        clips_events = self.clips.events(cog_teams=[t.name for t in self.teams.values()], max_steps=self.max_steps)
        weather_events = self.weather.events(max_steps=self.max_steps)
        overlap = set(clips_events) & set(weather_events)
        if overlap:
            raise ValueError(f"Overlapping event keys between clips and weather: {overlap}")
        return {**clips_events, **weather_events}
