"""Four Corners variant: 1-4 teams in corner compounds."""

from __future__ import annotations

from typing import override

from pydantic import Field

from cogames.core import CoGameMissionVariant, Deps
from cogames.games.cogs_vs_clips.game.clips.clips import NoClipsVariant
from cogames.games.cogs_vs_clips.game.extractors import ExtractorsVariant
from cogames.games.cogs_vs_clips.game.teams.team import TeamConfig, TeamVariant
from cogames.games.cogs_vs_clips.missions.mission import CvCMission
from cogames.games.cogs_vs_clips.missions.terrain import CompoundLocation, MachinaTerrainVariant
from cogames.variants import ResolvedDeps
from mettagrid.config.game_value import SumGameValue, num_tagged, val, weighted_sum
from mettagrid.config.handler_config import Handler, allOf
from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.config.mutation import logStatToGame
from mettagrid.config.reward_config import reward
from mettagrid.mapgen.scenes.compound import CompoundConfig

TEAM_COLORS = ["red", "blue", "green", "yellow"]
CORNER_LOCS: list[CompoundLocation] = ["nw", "ne", "sw", "se"]


class FourScoreVariant(CoGameMissionVariant):
    """Set up 1-4 teams with corner compounds."""

    name: str = "four_score"
    description: str = "Multi-team corner compounds."
    num_teams: int = Field(default=4, ge=1, le=4)

    @override
    def dependencies(self) -> Deps:
        return Deps(
            required=[
                TeamVariant,
                NoClipsVariant,
                MachinaTerrainVariant,
                ExtractorsVariant,
            ]
        )

    @override
    def configure(self, deps: ResolvedDeps) -> None:
        # Set up teams.
        team_v = deps.required(TeamVariant)
        team_v.teams = {
            f"cogs_{TEAM_COLORS[i]}": TeamConfig(
                name=f"cogs_{TEAM_COLORS[i]}",
                short_name=TEAM_COLORS[i][:3],
            )
            for i in range(self.num_teams)
        }

        # Reduce extractor stock for four-score so corners do not start too rich.
        deps.required(ExtractorsVariant).initial_amount = 100

        # Configure terrain with corner compounds.
        terrain = deps.required(MachinaTerrainVariant)
        terrain.map_width = 120
        terrain.map_height = 120
        terrain.building_coverage_scale = 1.5
        terrain.compound_placements = [
            (
                CORNER_LOCS[i],
                CompoundConfig(
                    hub_object=f"{TEAM_COLORS[i][:3]}:hub",
                    corner_bundle="extractors",
                    cross_objects=["junction", "", "", ""],
                    cross_distance=4,
                    outer_clearance=1,
                    spawn_symbol=f"agent.team_{i}",
                ),
            )
            for i in range(self.num_teams)
        ]

    @override
    def modify_env(self, mission: CvCMission, env: MettaGridConfig) -> None:
        team_v = mission.required_variant(TeamVariant)
        seen_team_names: set[str] = set()
        held_junction_stats: list[SumGameValue] = []

        # Per-team junction rewards and held-tick stats.
        for agent in env.game.agents:
            team_name = team_v.team_name(agent.team_id)
            assert team_name is not None, f"agent team_id={agent.team_id} has no team name"
            held_junction_values = [num_tagged(f"net:{team_name}"), val(-1.0)]
            held_junctions = SumGameValue(values=held_junction_values)
            if team_name not in seen_team_names:
                held_junction_stats.append(held_junctions)
            agent.rewards["aligned_junction_held"] = reward(
                held_junction_values,
                weight=1.0 / mission.max_steps,
                per_tick=True,
            )
            if team_name in seen_team_names:
                continue
            handler = Handler(
                name=f"aligned_junction_held_{team_name}",
                mutations=[logStatToGame(f"{team_name}/aligned.junction.held", source=held_junctions)],
            )
            env.game.on_tick = allOf([env.game.on_tick, handler])
            seen_team_names.add(team_name)

        if held_junction_stats:
            avg_held_junctions = weighted_sum(
                [(1.0 / len(held_junction_stats), held_stat) for held_stat in held_junction_stats]
            )
            handler = Handler(
                name="aligned_junction_held_four_score_avg",
                mutations=[logStatToGame("four_score/aligned.junction.held.avg", source=avg_held_junctions)],
            )
            env.game.on_tick = allOf([env.game.on_tick, handler])
