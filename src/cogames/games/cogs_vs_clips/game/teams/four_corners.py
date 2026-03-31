"""Four Corners variant: replace single hub with 1-4 corner compounds, each with its own team."""

from __future__ import annotations

from typing import override

from pydantic import Field

from cogames.core import CoGameMissionVariant, Deps
from cogames.games.cogs_vs_clips.game.clips.clips import NoClipsVariant
from cogames.games.cogs_vs_clips.game.teams.gear_stations import TeamGearStationsVariant
from cogames.games.cogs_vs_clips.game.teams.team import TeamConfig, TeamVariant
from cogames.games.cogs_vs_clips.game.terrain.four_corners import (
    _cvc_compound_template,
    build_four_corners_map,
    find_four_corners,
)
from cogames.games.cogs_vs_clips.missions.mission import CvCMission
from cogames.games.cogs_vs_clips.missions.terrain import find_machina_arena
from cogames.variants import ResolvedDeps
from mettagrid.config.game_value import num_tagged, val
from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.config.render_config import RenderAsset
from mettagrid.config.reward_config import reward

TEAM_COLORS = ["red", "blue", "green", "yellow"]


class FourCornersVariant(CoGameMissionVariant):
    """Replace single center hub with 1-4 corner compounds, each with its own team."""

    name: str = "four_corners"
    description: str = "Multi-team corner compounds with colored sprites."
    num_teams: int = Field(default=4, ge=1, le=4)

    @override
    def dependencies(self) -> Deps:
        return Deps(required=[TeamVariant, NoClipsVariant], optional=[TeamGearStationsVariant])

    @override
    def configure(self, deps: ResolvedDeps) -> None:
        team_v = deps.required(TeamVariant)
        team_v.teams = {
            f"cogs_{TEAM_COLORS[i]}": TeamConfig(
                name=f"cogs_{TEAM_COLORS[i]}",
                short_name=TEAM_COLORS[i][:3],
            )
            for i in range(self.num_teams)
        }

    @override
    def modify_env(self, mission: CvCMission, env: MettaGridConfig) -> None:
        team_v = mission.required_variant(TeamVariant)
        teams = [t for t in team_v.teams.values() if t.num_agents > 0]

        # Collect hub stations from the existing map before replacing it.
        existing_arena = find_machina_arena(env.game.map_builder)
        hub_stations = list(existing_arena.hub.stations) if existing_arena else []

        # Replace the map builder with a 120x120 four-corners layout.
        env.game.map_builder = build_four_corners_map(len(teams), max(t.num_agents for t in teams))

        # Configure the corner compounds with team-specific objects.
        arena = find_machina_arena(env.game.map_builder)
        if arena is None:
            return
        fcc = find_four_corners(arena)
        if fcc is None:
            return

        fcc.compound = _cvc_compound_template()
        fcc.hub_objects = [f"{t.short_name}:hub" for t in teams]
        fcc.spawn_symbols = [f"agent.team_{t.team_id}" for t in teams]
        fcc.stations_per_compound = [[s for s in hub_stations if s.startswith(f"{t.short_name}:")] for t in teams]
        fcc.num_compounds = len(teams)
        fcc.spawn_count = max(t.num_agents for t in teams)

        # Per-team colored agent sprites.
        env.game.render.assets["agent"] = [
            RenderAsset(asset=f"agent.{TEAM_COLORS[i]}", tags=[t.team_tag()]) for i, t in enumerate(teams)
        ] + [RenderAsset(asset="agent")]

        # Per-team junction rewards.
        for agent in env.game.agents:
            team_name = team_v.team_name(agent.team_id)
            if team_name is not None:
                agent.rewards["aligned_junction_held"] = reward(
                    [num_tagged(f"net:{team_name}"), val(-1.0)],
                    weight=1.0 / mission.max_steps,
                    per_tick=True,
                )
