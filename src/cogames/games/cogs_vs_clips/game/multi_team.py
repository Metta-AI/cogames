"""MultiTeam variant: splits the map into separate team instances."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from pydantic import Field

from cogames.core import CoGameMissionVariant, Deps
from cogames.games.cogs_vs_clips.game.teams import TeamVariant
from cogames.games.cogs_vs_clips.missions.terrain import find_machina_arena
from cogames.variants import ResolvedDeps
from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.map_builder.map_builder import MapBuilderConfig
from mettagrid.mapgen.mapgen import MapGen

if TYPE_CHECKING:
    from cogames.games.cogs_vs_clips.missions.mission import CvCMission

GEAR = ["aligner", "scrambler", "miner", "scout"]


def _set_spawn_count(builder: MapBuilderConfig, spawn_count: int) -> None:
    """Set spawn_count in nested MachinaArenaConfig so map cells match agent count."""
    arena = find_machina_arena(builder)
    if arena is not None:
        arena.spawn_count = spawn_count


class MultiTeamVariant(CoGameMissionVariant):
    """Split the map into multiple team instances, each with their own hub and resources."""

    name: str = "multi_team"
    description: str = "Split map into separate team instances with independent hubs."
    num_teams: int = Field(default=2, ge=2, le=2, description="Number of teams (max 2 supported)")

    @override
    def dependencies(self) -> Deps:
        return Deps(required=[TeamVariant])

    @override
    def configure(self, deps: ResolvedDeps) -> None:
        team_v = deps.required(TeamVariant)
        team = next(iter(team_v.teams.values()))
        team_v.teams = {
            name: team.model_copy(update={"name": name, "short_name": name})
            for name in ["cogs_green", "cogs_blue"][: self.num_teams]
        }

    @staticmethod
    def _normalize(key: str, team_prefixes: set[str]) -> str:
        """Replace a team-specific prefix with the template 'c:' prefix."""
        for prefix in team_prefixes:
            if key.startswith(prefix):
                return f"c:{key[len(prefix) :]}"
        return key

    @override
    def modify_env(self, mission: CvCMission, env: MettaGridConfig) -> None:
        team_v = mission.required_variant(TeamVariant)
        agents_per_team = max(t.num_agents for t in team_v.teams.values())
        original_builder = env.game.map_builder
        if isinstance(original_builder, MapGen.Config):
            original_builder = original_builder.model_copy(deep=True)
            original_builder.border_width = 1
            _set_spawn_count(original_builder, agents_per_team)

        # Normalize hub object names from team-specific prefixes (e.g.
        # "cogs_green:hub") back to the template "c:" prefix so that
        # instance_object_remap can remap them per-instance.
        team_prefixes = {f"{t.short_name}:" for t in team_v.teams.values()}
        arena = find_machina_arena(original_builder) if isinstance(original_builder, MapGen.Config) else None
        if arena is not None:
            arena.hub.hub_object = self._normalize(arena.hub.hub_object, team_prefixes)
            seen: set[str] = set()
            normalized: list[str] = []
            for key in arena.hub.stations:
                key = self._normalize(key, team_prefixes)
                if key not in seen:
                    seen.add(key)
                    normalized.append(key)
            arena.hub.stations = normalized

        # Build remap: c:hub plus every c:-prefixed station and gear item.
        object_remap: dict[str, str] = {"c:hub": "{instance_name}:hub"}
        object_remap.update({f"c:{g}": f"{{instance_name}}:{g}" for g in GEAR})
        if arena is not None:
            for key in arena.hub.stations:
                if key.startswith("c:"):
                    object_remap.setdefault(key, f"{{instance_name}}:{key[2:]}")

        env.game.map_builder = MapGen.Config(
            instance=original_builder,
            instances=self.num_teams,
            set_team_by_instance=True,
            instance_names=[t.short_name for t in team_v.teams.values()],
            instance_object_remap=object_remap,
            border_width=0,
            instance_border_width=0,
            instance_border_clear_radius=3,
        )
