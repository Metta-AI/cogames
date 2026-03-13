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
    add_clips_ships_to_map_config,
    clips_ship_map_names_in_map_config,
)
from cogames.games.cogs_vs_clips.game.junction import JunctionVariant
from cogames.games.cogs_vs_clips.game.multi_team import MultiTeamVariant
from cogames.games.cogs_vs_clips.game.teams import TeamConfig
from cogames.games.cogs_vs_clips.game.teams.team import TeamVariant
from cogames.variants import ResolvedDeps
from mettagrid.config.event_config import EventConfig, periodic
from mettagrid.config.filter import hasTag, hasTagPrefix, isNear, isNot, maxDistance
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

    def _ship_map_names(self) -> list[str]:
        return [f"{self.short_name}:ship", *[f"{self.short_name}:ship:{idx}" for idx in range(4)]]

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

            events[align_key] = EventConfig(
                name=align_key,
                target_query=query(
                    typeTag("junction"),
                    filters=[
                        isNot(hasTagPrefix("team:")),
                        isNear(ship_frontier, radius=JUNCTION_ALIGN_DISTANCE),
                    ],
                ),
                timesteps=periodic(start=self.align_start, period=self.align_interval, end=align_end),
                mutations=[
                    addTag(self.team_tag()),
                    addTag(self.net_tag()),
                    addTag(ship_map_name),
                    recomputeMaterializedQuery("net:"),
                ],
                max_targets=1,
            )
            events[scramble_key] = EventConfig(
                name=scramble_key,
                target_query=query(
                    typeTag("junction"),
                    filters=[
                        hasTagPrefix("team:"),
                        isNot(hasTag(self.team_tag())),
                        isNear(ship_frontier, radius=self.scramble_radius),
                    ],
                ),
                timesteps=periodic(start=self.scramble_start, period=self.scramble_interval, end=scramble_end),
                mutations=[
                    removeTagPrefix("net:"),
                    removeTag(ship_map_name),
                    logActorAgentStat("junction.scrambled_by_clips"),
                    recomputeMaterializedQuery("net:"),
                ],
                max_targets=1,
            )
        return events

    def ship_stations(self) -> dict[str, GridObjectConfig]:
        ship = CvCShipConfig()
        stations: dict[str, GridObjectConfig] = {}
        for ship_map_name in self._ship_map_names():
            cfg = ship.station_cfg(team=self, map_name=ship_map_name)
            cfg.tags = [*cfg.tags, ship_map_name]
            stations[ship_map_name] = cfg
        return stations


class ClipsVariant(CoGameMissionVariant):
    """Add clips: a non-player faction that spreads via events and pressures cogs."""

    name: str = "clips"
    description: str = "Add clips faction with ships that spread across junctions."
    num_ships: int = Field(default=4, description="Number of clips ships to place on the map")
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
        env.game.map_builder = add_clips_ships_to_map_config(env.game.map_builder, self.num_ships)  # type: ignore[assignment]

        if self.clips is None or not isinstance(self.clips, ClipsConfig):
            return

        for name, station in self.clips.ship_stations().items():
            env.game.objects.setdefault(name, station)
            env.game.render.symbols.setdefault(name, "🚀")

        clips_events = self.clips.events(
            max_steps=env.game.max_steps,
            map_builder=env.game.map_builder,
        )
        env.game.events.update(clips_events)
