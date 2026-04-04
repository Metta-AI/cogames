"""Team gear stations: per-team gear stations that charge the hub."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from cogames.core import CoGameMissionVariant, Deps
from cogames.games.cogs_vs_clips.game.gear import GearVariant
from cogames.games.cogs_vs_clips.game.teams.hub import CvCHubConfig, TeamHubVariant
from cogames.games.cogs_vs_clips.game.teams.team import TeamVariant
from cogames.games.cogs_vs_clips.missions.terrain import find_machina_arena
from mettagrid.config.filter import sharedTagPrefix
from mettagrid.config.handler_config import (
    ClearInventoryMutation,
    EntityTarget,
    Handler,
    actorHas,
    queryDelta,
    updateActor,
)
from mettagrid.config.mettagrid_config import GridObjectConfig, MettaGridConfig
from mettagrid.config.render_config import RenderAsset

if TYPE_CHECKING:
    from cogames.games.cogs_vs_clips.game.teams.team import TeamConfig
    from cogames.games.cogs_vs_clips.missions.mission import CvCMission


DEFAULT_TEAM_GEAR_SYMBOLS = {
    "aligner": "🔗",
    "scrambler": "🌀",
    "miner": "⛏️",
    "scout": "🔭",
}


class TeamGearStationsVariant(CoGameMissionVariant):
    """Create per-team gear stations that charge costs from the team hub."""

    name: str = "team_gear_stations"
    description: str = "Per-team gear stations with hub-based costs."

    @override
    def dependencies(self) -> Deps:
        return Deps(required=[TeamHubVariant, GearVariant, TeamVariant])

    @override
    def modify_env(self, mission: CvCMission, env: MettaGridConfig) -> None:
        gear = mission.required_variant(GearVariant)
        team_v = mission.required_variant(TeamVariant)
        station_keys: list[str] = []
        for team in (t for t in team_v.teams.values() if t.num_agents > 0):
            for item_name in gear.items:
                symbol = gear.station_symbols.get(item_name, DEFAULT_TEAM_GEAR_SYMBOLS.get(item_name, "📦"))
                self._add_station(env, team, item_name, symbol, gear.station_costs.get(item_name))
                station_keys.append(f"{team.short_name}:{item_name}")

        arena = find_machina_arena(env.game.map_builder)
        if arena is not None:
            existing = set(arena.hub.stations)
            arena.hub.stations.extend(k for k in station_keys if k not in existing)

    @staticmethod
    def _add_station(
        env: MettaGridConfig,
        team: TeamConfig,
        gear_type: str,
        symbol: str,
        cost: dict[str, int] | None = None,
    ) -> None:
        key = f"{team.short_name}:{gear_type}"
        station = env.game.objects.setdefault(
            key,
            GridObjectConfig(
                name=gear_type,
                map_name=key,
                tags=[f"team:{team.name}"],
            ),
        )
        env.game.render.symbols[key] = symbol
        if not isinstance(station, GridObjectConfig):
            return

        hq = CvCHubConfig.hub_query(team)
        change_filters: list = [sharedTagPrefix("team:")]
        change_mutations: list = [ClearInventoryMutation(target=EntityTarget.ACTOR, limit_name="gear")]
        if cost:
            change_filters.extend(CvCHubConfig.hub_has(team, cost))
            change_mutations.append(queryDelta(hq, {k: -v for k, v in cost.items()}))
        change_mutations.append(updateActor({gear_type: 1}))

        station.on_use_handlers.update(
            {
                "keep_gear": Handler(filters=[sharedTagPrefix("team:"), actorHas({gear_type: 1})], mutations=[]),
                "change_gear": Handler(filters=change_filters, mutations=change_mutations),
            }
        )
        env.game.render.assets[key] = [RenderAsset(asset=f"{gear_type}_station")]
