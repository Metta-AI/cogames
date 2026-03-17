"""Machina-1 mission definitions.

Composes clips, day/night, HP damage, and gear costs into mission factories.
"""

from __future__ import annotations

from typing import override

from pydantic import Field

from cogames.core import CoGameMissionVariant, Deps
from cogames.games.cogs_vs_clips.game.cargo import CargoLimitVariant
from cogames.games.cogs_vs_clips.game.clips.clips import ClipsVariant
from cogames.games.cogs_vs_clips.game.damage import DamageVariant
from cogames.games.cogs_vs_clips.game.days import DaysVariant
from cogames.games.cogs_vs_clips.game.elements import ElementsVariant
from cogames.games.cogs_vs_clips.game.energy import EnergyVariant
from cogames.games.cogs_vs_clips.game.extractors import ExtractorsVariant
from cogames.games.cogs_vs_clips.game.gear import GearVariant
from cogames.games.cogs_vs_clips.game.heart import HeartVariant
from cogames.games.cogs_vs_clips.game.roles.aligner import AlignerVariant
from cogames.games.cogs_vs_clips.game.roles.miner import MinerVariant
from cogames.games.cogs_vs_clips.game.roles.scout import ScoutVariant
from cogames.games.cogs_vs_clips.game.roles.scrambler import ScramblerVariant
from cogames.games.cogs_vs_clips.game.teams.gear_stations import TeamGearStationsVariant
from cogames.games.cogs_vs_clips.game.teams.hub_observations import HubObservationsVariant
from cogames.games.cogs_vs_clips.game.teams.junction import TeamJunctionVariant
from cogames.games.cogs_vs_clips.game.teams.team import TeamVariant
from cogames.games.cogs_vs_clips.game.territory.damage_strangers import DamageStrangersVariant
from cogames.games.cogs_vs_clips.game.territory.heal_team import HealTeamVariant
from cogames.games.cogs_vs_clips.game.territory.territory import TerritoryVariant
from cogames.games.cogs_vs_clips.game.vibes import VibesVariant
from cogames.games.cogs_vs_clips.missions.mission import CvCMission
from cogames.games.cogs_vs_clips.missions.terrain import (
    MachinaArenaConfig,
    SequentialMachinaArena,
)
from cogames.variants import ResolvedDeps
from mettagrid.config.game_value import num_tagged, val
from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.config.reward_config import reward
from mettagrid.mapgen.mapgen import MapGen, MapGenConfig
from mettagrid.mapgen.scenes.building_distributions import DistributionConfig, DistributionType
from mettagrid.mapgen.scenes.compound import CompoundConfig

MACHINA_1_MAP_BUILDER = MapGen.Config(
    width=88,
    height=88,
    instance=SequentialMachinaArena.Config(
        spawn_count=20,
        map_corner_offset=1,
    ),
)


def _cvc_compound_config() -> CompoundConfig:
    return CompoundConfig(
        hub_object="empty",
        corner_bundle="none",
        cross_bundle="none",
        cross_distance=7,
    )


def _build_machina1_map_builder(spawn_count: int) -> MapGenConfig:
    map_builder = MACHINA_1_MAP_BUILDER.model_copy(deep=True)
    instance = map_builder.instance
    assert instance is not None
    assert isinstance(instance, MachinaArenaConfig)
    existing_building_distributions = instance.building_distributions or {}
    existing_building_distributions = {
        k: (DistributionConfig.model_validate(v) if isinstance(v, dict) else v)
        for k, v in existing_building_distributions.items()
    }
    return map_builder.model_copy(
        update={
            "instance": instance.model_copy(
                update={
                    "spawn_count": spawn_count,
                    "hub": _cvc_compound_config(),
                    "building_distributions": {
                        **existing_building_distributions,
                        "junction": DistributionConfig(type=DistributionType.POISSON),
                    },
                }
            ),
        }
    )


GEAR_COSTS: dict[str, dict[str, int]] = {
    "aligner": {"carbon": 3, "oxygen": 1, "germanium": 1, "silicon": 1},
    "scrambler": {"carbon": 1, "oxygen": 3, "germanium": 1, "silicon": 1},
    "miner": {"carbon": 1, "oxygen": 1, "germanium": 3, "silicon": 1},
    "scout": {"carbon": 1, "oxygen": 1, "germanium": 1, "silicon": 3},
}


class CvCMachina1Variant(CoGameMissionVariant):
    """Cross-configures machina1 sub-variants (gear costs, junction costs, damage, rewards)."""

    name: str = "machina_1"
    description: str = "Clips + day/night cycle + HP damage + gear costs."

    @override
    def dependencies(self) -> Deps:
        return Deps(
            required=[
                VibesVariant,
                TeamVariant,
                HubObservationsVariant,
                TerritoryVariant,
                ElementsVariant,
                HeartVariant,
                TeamJunctionVariant,
                DamageVariant,
                EnergyVariant,
                CargoLimitVariant,
                ExtractorsVariant,
                GearVariant,
                AlignerVariant,
                ScramblerVariant,
                MinerVariant,
                ScoutVariant,
                ClipsVariant,
                DaysVariant,
                TeamGearStationsVariant,
                DamageStrangersVariant,
                HealTeamVariant,
            ]
        )

    @override
    def configure(self, deps: ResolvedDeps) -> None:
        elements = deps.required(ElementsVariant).elements

        heart = deps.required(HeartVariant)
        heart.cost = {e: 7 for e in elements}

        tj = deps.required(TeamJunctionVariant)
        tj.align_cost = {"heart": 1}
        tj.scramble_cost = {"heart": 1}

        deps.required(GearVariant).station_costs = GEAR_COSTS

    @override
    def modify_env(self, mission: CvCMission, env: MettaGridConfig) -> None:
        team_v = mission.required_variant(TeamVariant)
        for agent in env.game.agents:
            team_name = team_v.team_name(agent.team_id)
            if team_name is not None:
                # net:* includes the team's hub, so subtract the root node and
                # reward only held junctions.
                agent.rewards["aligned_junction_held"] = reward(
                    [num_tagged(f"net:{team_name}"), val(-1.0)],
                    weight=1.0 / mission.max_steps,
                    per_tick=True,
                )


class MachinaOneMission(CvCMission):
    """Machina-1 mission: clips, day/night, HP damage, gear costs, junction control."""

    name: str = "machina_1"
    description: str = "CvC Machina1 - compete to control junctions with gear abilities."
    map_builder: MapGenConfig = Field(default_factory=lambda: _build_machina1_map_builder(20))
    num_cogs: int = 8
    min_cogs: int = 1
    max_cogs: int = 20
    max_steps: int = 10000
    default_variant: str = "machina_1"
    sub_missions: list[str] = Field(default_factory=lambda: list(["solo", "clips", "auto_clips"]))


def make_machina1_map_builder(num_agents: int = 10) -> MapGenConfig:
    """Create a Machina-1 map builder with configurable agent count."""
    return _build_machina1_map_builder(num_agents)


def make_machina1_mission(num_agents: int = 10, max_steps: int = 10000) -> CvCMission:
    """Create a CvC mission with clips and weather (Machina1 layout)."""
    return MachinaOneMission(
        map_builder=_build_machina1_map_builder(num_agents),
        num_agents=num_agents,
        num_cogs=num_agents,
        min_cogs=num_agents,
        max_cogs=num_agents,
        max_steps=max_steps,
    )


# Aliases for backwards compatibility with envs/tournament code
make_cogsguard_mission = make_machina1_mission


def make_game(num_cogs: int = 2) -> MettaGridConfig:
    """Create a default CvC game configuration."""
    return make_cogsguard_mission(num_agents=num_cogs).make_env()
