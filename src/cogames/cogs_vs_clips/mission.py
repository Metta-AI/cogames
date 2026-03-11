from __future__ import annotations

from pydantic import Field

from cogames.cogs_vs_clips.clips import ClipsConfig
from cogames.cogs_vs_clips.cog import CogConfig, CogTeam
from cogames.cogs_vs_clips.config import CvCConfig
from cogames.cogs_vs_clips.junction import CvCJunctionConfig
from cogames.cogs_vs_clips.machina_1_variant import CvCMachina1Variant  # noqa: F401 - register variant
from cogames.cogs_vs_clips.render import render_config
from cogames.cogs_vs_clips.stations import (
    CvCExtractorConfig,
    CvCWallConfig,
)
from cogames.cogs_vs_clips.variants import NumCogsVariant
from cogames.cogs_vs_clips.weather import WeatherConfig
from cogames.core import (
    MAP_MISSION_DELIMITER,
    CoGameMission,
    CoGameMissionVariant,
    CoGameSite,
)
from cogames.variants import VariantRegistry
from mettagrid.config.action_config import (
    ActionsConfig,
    ChangeVibeActionConfig,
    MoveActionConfig,
    NoopActionConfig,
)
from mettagrid.config.filter import sharedTagPrefix
from mettagrid.config.game_value import QueryInventoryValue
from mettagrid.config.handler_config import Handler, updateTarget
from mettagrid.config.mettagrid_config import GameConfig, MettaGridConfig
from mettagrid.config.obs_config import GlobalObsConfig, ObsConfig
from mettagrid.config.query import query
from mettagrid.config.tag import typeTag
from mettagrid.config.territory_config import TerritoryConfig
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
    default_variant: str | None = Field(default="machina_1")
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

        Uses VariantRegistry to resolve dependencies and apply variants
        in topological order (e.g. energy -> solar -> days).

        Returns:
            MettaGridConfig ready for environment creation
        """
        # default_variant injects structural variants (energy→solar→days) that
        # every CvC mission needs but that shouldn't appear in user-facing labels.
        # It exists because these variants still read config from mission.cog /
        # mission.weather rather than owning their own fields. Once config ownership
        # migrates into the variants themselves, each variant becomes self-contained
        # and can be listed in `variants=[]` directly — at which point default_variant
        # can be removed.
        registry = VariantRegistry(list(self.variants))
        variant_names: list[str] = []
        if self.default_variant:
            variant_names.append(self.default_variant)
        variant_names.extend(v.name for v in self.variants)
        registry.run_configure(variant_names)

        team_objs = list(self.teams.values())
        for i, t in enumerate(team_objs):
            t.team_id = i
        self.clips.team_id = len(team_objs)

        all_teams = [*team_objs, self.clips]

        game = GameConfig(
            map_builder=self.map_builder(),
            max_steps=self.max_steps,
            num_agents=self.num_agents,
            resource_names=CvCConfig.RESOURCES,
            render=render_config(team_objs=team_objs),
            territories={
                "team_territory": TerritoryConfig(
                    tag_prefix="team:",
                    presence={
                        "heal": Handler(
                            filters=[sharedTagPrefix("team:")],
                            mutations=[updateTarget({"energy": 100, "hp": 100})],
                        ),
                    },
                ),
            },
            vibe_names=CvCConfig.VIBE_NAMES,
            obs=ObsConfig(
                global_obs=GlobalObsConfig(
                    obs={
                        f"team:{resource}": QueryInventoryValue(
                            query=query(typeTag("hub"), sharedTagPrefix("team:")),
                            item=resource,
                        )
                        for resource in CvCConfig.ELEMENTS
                    },
                    local_position=True,
                    last_action_move=True,
                ),
                aoe_mask=True,
            ),
            actions=ActionsConfig(
                move=MoveActionConfig(),
                noop=NoopActionConfig(),
                change_vibe=ChangeVibeActionConfig(vibes=CvCConfig.VIBES),
            ),
            agents=[
                self.cog.agent_config(team=t, max_steps=self.max_steps) for t in team_objs for _ in range(t.num_agents)
            ],
            objects={
                "wall": CvCWallConfig().station_cfg(),
                "junction": CvCJunctionConfig().station_cfg(teams=all_teams),
                **{
                    f"{resource}_extractor": CvCExtractorConfig(resource=resource).station_cfg()
                    for resource in CvCConfig.ELEMENTS
                },
                **{name: cfg for team in all_teams for name, cfg in team.stations().items()},
            },
            events=self.clips.events(max_steps=self.max_steps, map_builder=self.map_builder()),
            tags=[tag for t in all_teams for tag in t.all_tags()],
            materialize_queries=[mq for t in all_teams for mq in t.materialized_queries()],
            on_tick={k: v for t in all_teams for k, v in t.on_tick_handlers(CvCConfig.RESOURCES).items()},
        )

        env = MettaGridConfig(game=game)
        # Precaution - copy the env, in case the code above uses some global variable that we don't want to modify.
        # This allows variants to modify the env without copying it again.
        env = env.model_copy(deep=True)
        env.label = self.full_name()

        registry.apply_to_env(self, env)

        for variant in self.variants:
            env.label += f".{variant.name}"

        return env
