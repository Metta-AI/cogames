"""Clips: non-player faction that spreads via events.

Clips are a non-player faction that gradually takes over neutral junctions.
These events create the spreading/scrambling behavior that pressures players.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, override

from pydantic import Field

from cogames.core import CoGameMissionVariant, Deps
from cogames.games.cogs_vs_clips.game.clips.ship import (
    CvCShipConfig,
    clips_ship_map_names_in_map_config,
    set_clips_ships_in_map_config,
)
from cogames.games.cogs_vs_clips.game.junction import JunctionVariant
from cogames.games.cogs_vs_clips.game.multi_team import MultiTeamVariant
from cogames.games.cogs_vs_clips.game.teams import TeamConfig
from cogames.games.cogs_vs_clips.game.teams.team import TeamVariant
from cogames.variants import ResolvedDeps
from mettagrid.config.event_config import EventConfig, periodic
from mettagrid.config.filter import AnyFilter, hasTag, hasTagPrefix, isNear, isNot, maxDistance
from mettagrid.config.mettagrid_config import GridObjectConfig, MettaGridConfig
from mettagrid.config.mutation import (
    addTag,
    logActorAgentStat,
    recomputeMaterializedQuery,
    removeTag,
    removeTagPrefix,
)
from mettagrid.config.query import ClosureQuery, Query, query
from mettagrid.config.tag import typeTag
from mettagrid.map_builder.map_builder import AnyMapBuilderConfig

JUNCTION_ALIGN_DISTANCE = 15

if TYPE_CHECKING:
    from cogames.games.cogs_vs_clips.missions.mission import CvCMission


class ClipsConfig(TeamConfig):
    """Configuration for clips behavior in CvC game mode."""

    name: str = Field(default="clips")
    short_name: str = Field(default="clips")
    num_agents: int = Field(default=0, ge=0)
    disabled: bool = Field(default=False)

    initial_clips_start: int = Field(default=10)
    initial_clips_spots: int = Field(default=1)

    scramble_start: int = Field(default=50)
    scramble_interval: int = Field(default=70)
    scramble_radius: int = Field(default=JUNCTION_ALIGN_DISTANCE)
    scramble_end: Optional[int] = Field(default=None)

    align_start: int = Field(default=100)
    align_interval: int = Field(default=70)
    align_end: Optional[int] = Field(default=None)

    presence_end: Optional[int] = Field(default=None)
    greedy_expand_from_ships: bool = Field(default=True)
    greedy_max_search_radius: int = Field(default=120, ge=1)
    angry_target_enemy_hub: bool = Field(default=False)

    def ship_query(self) -> Query:
        return query(typeTag("ship"), hasTag(self.team_tag()))

    def events(
        self,
        max_steps: int,
        map_builder: AnyMapBuilderConfig,
    ) -> dict[str, EventConfig]:
        if self.disabled:
            return {}
        ship_map_names = clips_ship_map_names_in_map_config(map_builder)
        ship_count = len(ship_map_names)
        if ship_count <= 0:
            return {}

        scramble_end = max_steps if self.scramble_end is None else min(self.scramble_end, max_steps)
        align_end = max_steps if self.align_end is None else min(self.align_end, max_steps)

        events: dict[str, EventConfig] = {}
        max_search_radius = max(1, self.greedy_max_search_radius)

        def add_greedy_event_chain(
            *,
            base_name: str,
            timesteps: list[int],
            target_filters: list[AnyFilter],
            center_query: Query,
            mutations: list,
        ) -> None:
            next_event_name: str | None = None
            for radius in range(max_search_radius, 0, -1):
                radius_name = base_name if radius == 1 else f"{base_name}_r{radius}"
                filters = [*target_filters, isNear(center_query, radius=radius)]
                events[radius_name] = EventConfig(
                    name=radius_name,
                    target_query=query(typeTag("junction"), filters=filters),
                    timesteps=timesteps if radius == 1 else [],
                    mutations=mutations,
                    max_targets=1,
                    fallback=next_event_name,
                )
                next_event_name = radius_name

        for lane_idx, ship_map_name in enumerate(ship_map_names):
            suffix = "" if lane_idx == 0 else f"_s{lane_idx}"
            align_key = f"neutral_to_clips{suffix}"
            scramble_key = f"cogs_to_neutral{suffix}"
            ship_query = query(
                typeTag("ship"),
                filters=[hasTag(self.team_tag()), hasTag(ship_map_name)],
            )
            ship_frontier = ClosureQuery(
                source=ship_query,
                candidates=query(typeTag("junction"), hasTag(ship_map_name)),
                edge_filters=[maxDistance(JUNCTION_ALIGN_DISTANCE)],
            )
            align_filters = [
                isNot(hasTagPrefix("team:")),
                isNear(ship_frontier, radius=JUNCTION_ALIGN_DISTANCE),
            ]
            scramble_filters = [
                hasTagPrefix("team:"),
                isNot(hasTag(self.team_tag())),
                isNear(ship_frontier, radius=self.scramble_radius),
            ]
            align_timesteps = periodic(start=self.align_start, period=self.align_interval, end=align_end)
            scramble_timesteps = periodic(start=self.scramble_start, period=self.scramble_interval, end=scramble_end)
            align_mutations = [
                addTag(self.team_tag()),
                addTag(self.net_tag()),
                addTag(ship_map_name),
                recomputeMaterializedQuery("net:"),
            ]
            scramble_mutations = [
                removeTagPrefix("net:"),
                removeTag(ship_map_name),
                logActorAgentStat("junction.scrambled_by_clips"),
                recomputeMaterializedQuery("net:"),
            ]
            enemy_hub_query = query(
                typeTag("hub"),
                filters=[
                    hasTagPrefix("team:"),
                    isNot(hasTag(self.team_tag())),
                ],
            )

            if self.greedy_expand_from_ships:
                add_greedy_event_chain(
                    base_name=align_key,
                    timesteps=align_timesteps,
                    target_filters=align_filters,
                    center_query=ship_query,
                    mutations=align_mutations,
                )
                add_greedy_event_chain(
                    base_name=scramble_key,
                    timesteps=scramble_timesteps,
                    target_filters=scramble_filters,
                    center_query=ship_query,
                    mutations=scramble_mutations,
                )
                continue

            if self.angry_target_enemy_hub:
                add_greedy_event_chain(
                    base_name=align_key,
                    timesteps=align_timesteps,
                    target_filters=align_filters,
                    center_query=enemy_hub_query,
                    mutations=align_mutations,
                )
                add_greedy_event_chain(
                    base_name=scramble_key,
                    timesteps=scramble_timesteps,
                    target_filters=scramble_filters,
                    center_query=enemy_hub_query,
                    mutations=scramble_mutations,
                )
                continue

            events[align_key] = EventConfig(
                name=align_key,
                target_query=query(
                    typeTag("junction"),
                    filters=align_filters,
                ),
                timesteps=align_timesteps,
                mutations=align_mutations,
                max_targets=1,
            )
            events[scramble_key] = EventConfig(
                name=scramble_key,
                target_query=query(
                    typeTag("junction"),
                    filters=scramble_filters,
                ),
                timesteps=scramble_timesteps,
                mutations=scramble_mutations,
                max_targets=1,
            )
        return events

    def ship_stations(self, map_builder: AnyMapBuilderConfig) -> dict[str, GridObjectConfig]:
        ship = CvCShipConfig()
        stations: dict[str, GridObjectConfig] = {}
        ship_map_names = list(dict.fromkeys(clips_ship_map_names_in_map_config(map_builder)))
        for ship_map_name in ship_map_names:
            cfg = ship.station_cfg(team=self, map_name=ship_map_name)
            cfg.tags = [*cfg.tags, ship_map_name]
            stations[ship_map_name] = cfg
        return stations


class ClipsVariant(CoGameMissionVariant):
    """Add clips: a non-player faction that spreads via events and pressures cogs."""

    name: str = "clips"
    description: str = "Add clips faction with ships that spread across junctions."
    num_ships: Optional[int] = Field(
        default=None,
        description="Resolved clips ship count. Defaults to the mission's built-in ship layout.",
    )
    clips_config: ClipsConfig = Field(default_factory=ClipsConfig)
    clips: ClipsConfig | None = Field(default=None, exclude=True)

    @override
    def dependencies(self) -> Deps:
        return Deps(required=[TeamVariant, JunctionVariant], optional=[MultiTeamVariant])

    @override
    def configure(self, deps: ResolvedDeps) -> None:
        team_v = deps.required(TeamVariant)
        self.clips = self.clips_config.model_copy(deep=True)
        team_v.teams[self.clips.name] = self.clips

    @override
    def modify_env(self, mission: CvCMission, env: MettaGridConfig) -> None:
        env.game.map_builder = set_clips_ships_in_map_config(env.game.map_builder, self.num_ships)  # type: ignore[assignment]

        if self.clips is None or not isinstance(self.clips, ClipsConfig):
            return

        for name, station in self.clips.ship_stations(env.game.map_builder).items():
            env.game.objects.setdefault(name, station)
            env.game.render.symbols.setdefault(name, "🚀")

        clips_events = self.clips.events(
            max_steps=env.game.max_steps,
            map_builder=env.game.map_builder,
        )
        env.game.events.update(clips_events)


class GreedyClipsVariant(CoGameMissionVariant):
    """Target the nearest valid junction to each clips ship."""

    name: str = "greedy_clips"
    description: str = "Clips spread from each ship by always selecting the nearest valid junction to that ship center."

    @override
    def dependencies(self) -> Deps:
        return Deps(required=[ClipsVariant])

    @override
    def configure(self, deps: ResolvedDeps) -> None:
        clips_v = deps.required(ClipsVariant)
        assert clips_v.clips is not None and isinstance(clips_v.clips, ClipsConfig)
        clips_v.clips.greedy_expand_from_ships = True
        clips_v.clips.scramble_radius = JUNCTION_ALIGN_DISTANCE


class AngryClipsVariant(CoGameMissionVariant):
    """Target frontier junctions nearest to the enemy hub."""

    name: str = "angry_clips"
    description: str = "Clips blitz toward the cogs hub by taking frontier junctions nearest to enemy hubs first."
    initial_clips_start: int = 400
    align_start: int = 400
    scramble_start: int = 400
    align_interval: int = 100
    scramble_interval: int = 100

    @override
    def dependencies(self) -> Deps:
        return Deps(required=[ClipsVariant])

    @override
    def configure(self, deps: ResolvedDeps) -> None:
        clips_v = deps.required(ClipsVariant)
        assert clips_v.clips is not None and isinstance(clips_v.clips, ClipsConfig)
        clips_v.clips.angry_target_enemy_hub = True
        clips_v.clips.greedy_expand_from_ships = False
        clips_v.clips.scramble_radius = JUNCTION_ALIGN_DISTANCE
        clips_v.clips.initial_clips_start = self.initial_clips_start
        clips_v.clips.align_start = self.align_start
        clips_v.clips.scramble_start = self.scramble_start
        clips_v.clips.align_interval = self.align_interval
        clips_v.clips.scramble_interval = self.scramble_interval


class NoClipsVariant(CoGameMissionVariant):
    """Set the resolved clips ship count to zero."""

    name: str = "no_clips"
    description: str = "Remove all clips ships so clips cannot spread."

    @override
    def dependencies(self) -> Deps:
        return Deps(required=[ClipsVariant])

    @override
    def configure(self, deps: ResolvedDeps) -> None:
        clips_variant = deps.required(ClipsVariant)
        clips_variant.num_ships = 0
