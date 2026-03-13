"""Tutorial missions — one mission with role sub-missions."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from pydantic import Field

from cogames.core import CoGameMissionVariant, Deps
from cogames.games.cogs_vs_clips.game.clips import ClipsConfig, ClipsVariant
from cogames.games.cogs_vs_clips.game.damage import DamageVariant
from cogames.games.cogs_vs_clips.game.days import DaysVariant
from cogames.games.cogs_vs_clips.game.elements import ElementsVariant
from cogames.games.cogs_vs_clips.game.extractors import ExtractorsVariant
from cogames.games.cogs_vs_clips.game.heart import HeartVariant
from cogames.games.cogs_vs_clips.game.roles.aligner import AlignerVariant
from cogames.games.cogs_vs_clips.game.roles.miner import MinerVariant
from cogames.games.cogs_vs_clips.game.roles.scout import ScoutVariant
from cogames.games.cogs_vs_clips.game.roles.scrambler import ScramblerVariant
from cogames.games.cogs_vs_clips.game.teams.gear_stations import TeamGearStationsVariant
from cogames.games.cogs_vs_clips.game.teams.hub_observations import HubObservationsVariant
from cogames.games.cogs_vs_clips.game.teams.junction import TeamJunctionVariant
from cogames.games.cogs_vs_clips.game.teams.team import TeamVariant
from cogames.games.cogs_vs_clips.game.territory.territory import TerritoryVariant
from cogames.games.cogs_vs_clips.game.vibes import VibesVariant
from cogames.games.cogs_vs_clips.missions.machina_1 import GEAR_COSTS
from cogames.games.cogs_vs_clips.missions.mission import CvCMission
from cogames.games.cogs_vs_clips.missions.terrain import MachinaArena
from cogames.games.cogs_vs_clips.train.reward_variants import (
    _apply_aligner,
    _apply_miner,
    _apply_scout,
    _apply_scrambler,
)
from cogames.variants import ResolvedDeps
from mettagrid.mapgen.mapgen import MapGen, MapGenConfig
from mettagrid.mapgen.scenes.compound import CompoundConfig

if TYPE_CHECKING:
    from mettagrid.config.mettagrid_config import MettaGridConfig


# TODO: unchecked variant
class CvCTutorialVariant(CoGameMissionVariant):
    """Machina1 variant without clips — for focused role tutorials."""

    name: str = "tutorial"
    description: str = "Machina1 mechanics without clips for tutorial practice."

    @override
    def dependencies(self) -> Deps:
        return Deps(
            required=[
                TeamVariant,
                VibesVariant,
                HubObservationsVariant,
                TerritoryVariant,
                ExtractorsVariant,
                AlignerVariant,
                ScramblerVariant,
                MinerVariant,
                ScoutVariant,
                DaysVariant,
                TeamGearStationsVariant,
                DamageVariant,
                ElementsVariant,
                HeartVariant,
                TeamJunctionVariant,
            ]
        )

    @override
    def configure(self, deps: ResolvedDeps) -> None:
        deps.required(TeamGearStationsVariant).costs = GEAR_COSTS

        elements = deps.required(ElementsVariant).elements
        deps.required(HeartVariant).cost = {e: 7 for e in elements}

        tj = deps.required(TeamJunctionVariant)
        tj.align_cost = {"heart": 1}
        tj.scramble_cost = {"heart": 1}


# TODO: unchecked variant
class AlignerRewardsVariant(CoGameMissionVariant):
    name: str = "aligner"
    description: str = "Learn aligner role - collect hearts, and align neutral junctions (no clips)."

    @override
    def modify_env(self, mission: CvCMission, env: MettaGridConfig) -> None:
        for agent_cfg in env.game.agents:
            rewards = dict(agent_cfg.rewards)
            _apply_aligner(rewards)
            agent_cfg.rewards = rewards


# TODO: unchecked variant
class MinerRewardsVariant(CoGameMissionVariant):
    name: str = "miner"
    description: str = "Learn miner role - resource extraction and deposits (no clips)."

    @override
    def modify_env(self, mission: CvCMission, env: MettaGridConfig) -> None:
        for agent_cfg in env.game.agents:
            rewards = dict(agent_cfg.rewards)
            _apply_miner(rewards)
            agent_cfg.rewards = rewards


# TODO: unchecked variant
class ScoutRewardsVariant(CoGameMissionVariant):
    name: str = "scout"
    description: str = "Learn scout role - exploration and visiting stale cells (no clips)."

    @override
    def modify_env(self, mission: CvCMission, env: MettaGridConfig) -> None:
        for agent_cfg in env.game.agents:
            rewards = dict(agent_cfg.rewards)
            _apply_scout(rewards)
            agent_cfg.rewards = rewards


class OverrunVariant(CoGameMissionVariant):
    name: str = "overrun"
    description: str = "All junctions start clips-aligned. No further clips spread."

    @override
    def dependencies(self) -> Deps:
        return Deps(required=[ClipsVariant])

    @override
    def configure(self, deps: ResolvedDeps) -> None:
        clips_v = deps.required(ClipsVariant)
        assert clips_v.clips is not None and isinstance(clips_v.clips, ClipsConfig)
        # Disable all clips events — junctions are tagged directly in modify_env.
        clips_v.clips.disabled = True

    @override
    def modify_env(self, mission: CvCMission, env: MettaGridConfig) -> None:
        clips_v = mission.required_variant(ClipsVariant)
        if clips_v.clips is None:
            return
        junction = env.game.objects.get("junction")
        if junction is None:
            return
        junction.tags.append(clips_v.clips.team_tag())
        junction.tags.append(clips_v.clips.net_tag())


# TODO: unchecked variant
class ScramblerRewardsVariant(CoGameMissionVariant):
    name: str = "scrambler"
    description: str = "Learn scrambler role - acquire scrambler gear and scramble enemy junctions."

    @override
    def dependencies(self) -> Deps:
        return Deps(required=[OverrunVariant])

    @override
    def configure(self, deps: ResolvedDeps) -> None:
        deps.required(OverrunVariant)

    @override
    def modify_env(self, mission: CvCMission, env: MettaGridConfig) -> None:
        for agent_cfg in env.game.agents:
            rewards = dict(agent_cfg.rewards)
            _apply_scrambler(rewards)
            agent_cfg.rewards = rewards


CVC_TUTORIAL_MAP_BUILDER = MapGen.Config(
    width=35,
    height=35,
    instance=MachinaArena.Config(
        spawn_count=4,
        building_coverage=0.05,
        hub=CompoundConfig(
            hub_object="c:hub",
            corner_bundle="extractors",
            cross_bundle="none",
            cross_distance=5,
            hub_width=15,
            hub_height=15,
            outer_clearance=2,
            stations=[
                "c:aligner",
                "c:scrambler",
                "c:miner",
                "c:scout",
            ],
        ),
    ),
)


TUTORIAL_SUB_MISSIONS = ["aligner", "miner", "scout", "scrambler"]


class TutorialMission(CvCMission):
    """Base tutorial: small map, no clips, role sub-missions for focused practice."""

    name: str = "tutorial"
    description: str = "Learn the basics of CvC: Roles, Resources, and Territory Control."
    map_builder: MapGenConfig = Field(default_factory=lambda: CVC_TUTORIAL_MAP_BUILDER.model_copy(deep=True))
    default_variant: str = "tutorial"
    num_cogs: int = 4
    min_cogs: int = 1
    max_cogs: int = 4
    num_agents: int = 4
    max_steps: int = 1000
    sub_missions: list[str] = Field(default_factory=lambda: list(TUTORIAL_SUB_MISSIONS))


def make_tutorial_mission() -> TutorialMission:
    return TutorialMission()
