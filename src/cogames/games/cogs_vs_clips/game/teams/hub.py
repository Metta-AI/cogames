"""Team hub variant: creates a hub station for each cog team."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from pydantic import Field

from cogames.core import CoGameMissionVariant, CvCStationConfig, Deps
from cogames.games.cogs_vs_clips.game.elements import ElementsVariant
from cogames.games.cogs_vs_clips.game.heart import HeartVariant
from cogames.games.cogs_vs_clips.game.teams.team import TeamConfig, TeamVariant
from cogames.games.cogs_vs_clips.missions.terrain import find_machina_arena
from mettagrid.config.filter import (
    GameValueFilter,
    HandlerTarget,
    actorHasAnyOf,
    hasTag,
    sharedTagPrefix,
)
from mettagrid.config.game_value import QueryInventoryValue
from mettagrid.config.handler_config import (
    Handler,
    queryDelta,
    queryDeposit,
    queryWithdraw,
    updateActor,
)
from mettagrid.config.mettagrid_config import (
    GridObjectConfig,
    InventoryConfig,
    MettaGridConfig,
    ResourceLimitsConfig,
)
from mettagrid.config.mutation.stats_mutation import logStatToGame
from mettagrid.config.query import Query, query
from mettagrid.config.tag import typeTag

if TYPE_CHECKING:
    from cogames.games.cogs_vs_clips.missions.mission import CvCMission


class CvCHubConfig(CvCStationConfig):
    """Hub station that accepts element deposits."""

    team: TeamConfig
    elements: list[str] = Field(default_factory=list)
    inventory: InventoryConfig = Field(default_factory=lambda: InventoryConfig(initial={}))

    @staticmethod
    def hub_query(team: TeamConfig) -> Query:
        """Query that finds the hub belonging to a team."""
        return query(typeTag("hub"), hasTag(team.team_tag()))

    @staticmethod
    def hub_has(team: TeamConfig, resources: dict[str, int]) -> list[GameValueFilter]:
        """Filters: team's hub has at least the given amount of each resource."""
        hq = CvCHubConfig.hub_query(team)
        return [
            GameValueFilter(
                target=HandlerTarget.TARGET,
                value=QueryInventoryValue(query=hq, item=resource),
                min=amount,
            )
            for resource, amount in resources.items()
        ]

    def station_cfg(self) -> GridObjectConfig:
        hq = self.hub_query(self.team)
        return GridObjectConfig(
            name="hub",
            map_name=f"{self.team.short_name}:hub",
            tags=[self.team.team_tag()],
            inventory=self.inventory,
            on_use_handlers={
                "deposit": Handler(
                    filters=[sharedTagPrefix("team:"), actorHasAnyOf(self.elements)],
                    mutations=[
                        queryDeposit(
                            hq,
                            {resource: 100 for resource in self.elements},
                            stat_prefix=f"{self.team.name}/",
                        ),
                    ],
                ),
            },
        )


class TeamHubVariant(CoGameMissionVariant):
    """Add a hub station for each team in the mission."""

    name: str = "team_hub"
    description: str = "Each team gets a hub station for deposits."
    initial_inventory: dict[str, InventoryConfig] = Field(
        default_factory=dict, description="Per-team hub inventory overrides, keyed by team name."
    )

    @staticmethod
    def _neg(recipe: dict[str, int]) -> dict[str, int]:
        return {k: -v for k, v in recipe.items()}

    @staticmethod
    def _hub_heart_handlers(team: TeamConfig, heart_cost: dict[str, int]) -> dict[str, Handler]:
        """Heart crafting and distribution handlers for a team's hub."""
        hq = CvCHubConfig.hub_query(team)
        return {
            "get_heart": Handler(
                filters=[sharedTagPrefix("team:"), *CvCHubConfig.hub_has(team, {"heart": 1})],
                mutations=[
                    queryWithdraw(hq, {"heart": 1}),
                    logStatToGame(f"{team.name}/heart.withdrawn"),
                ],
            ),
            "make_and_get_heart": Handler(
                filters=[sharedTagPrefix("team:"), *CvCHubConfig.hub_has(team, heart_cost)],
                mutations=[
                    queryDelta(hq, TeamHubVariant._neg(heart_cost)),
                    updateActor({"heart": 1}),
                    *[logStatToGame(f"{team.name}/{resource}.withdrawn") for resource in heart_cost],
                ],
            ),
        }

    @override
    def dependencies(self) -> Deps:
        return Deps(required=[TeamVariant, ElementsVariant])

    @override
    def modify_env(self, mission: CvCMission, env: MettaGridConfig) -> None:
        arena = find_machina_arena(env.game.map_builder)
        if arena is not None:
            team_v_for_hub = mission.required_variant(TeamVariant)
            team = next(iter(team_v_for_hub.teams.values()))
            arena.hub = arena.hub.model_copy(update={"hub_object": f"{team.short_name}:hub"})

        elements = mission.required_variant(ElementsVariant).elements
        heart = mission.optional_variant(HeartVariant)
        team_v = mission.required_variant(TeamVariant)

        for team in (t for t in team_v.teams.values() if t.num_agents > 0):
            default_initial: dict[str, int] = {element: team.num_agents * 3 for element in elements}
            if heart is not None:
                default_initial["heart"] = 5
            inventory = self.initial_inventory.get(
                team.name,
                InventoryConfig(
                    limits={"resources": ResourceLimitsConfig(base=10000, resources=elements)},
                    initial=default_initial,
                ),
            )
            hub = CvCHubConfig(team=team, elements=elements, inventory=inventory)
            map_name = f"{team.short_name}:hub"
            cfg = hub.station_cfg()
            if heart is not None:
                heart_cost = heart.cost or {}
                cfg.on_use_handlers.update(self._hub_heart_handlers(team, heart_cost))
            env.game.objects.setdefault(map_name, cfg)
