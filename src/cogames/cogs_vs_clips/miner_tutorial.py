"""Miner tutorial mission configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from cogames.cogs_vs_clips.clip_difficulty import EASY
from cogames.cogs_vs_clips.mission import CvCMission
from cogames.cogs_vs_clips.sites import COGSGUARD_ARENA
from cogames.cogs_vs_clips.team import CogTeam
from cogames.core import CoGameMissionVariant

if TYPE_CHECKING:
    from mettagrid.config.mettagrid_config import MettaGridConfig


class MinerRewardsVariant(CoGameMissionVariant):
    name: str = "miner_rewards"
    description: str = "Miner-focused reward shaping (gear, extraction, deposits)."

    @override
    def modify_env(self, mission: CvCMission, env: MettaGridConfig) -> None:
        from cogames.cogs_vs_clips.reward_variants import _apply_miner  # noqa: PLC0415

        for agent_cfg in env.game.agents:
            rewards = dict(agent_cfg.rewards)
            _apply_miner(rewards)
            agent_cfg.rewards = rewards


MinerTutorialMission = CvCMission(
    name="miner_tutorial",
    description="Learn miner role - resource extraction and deposits (no clips).",
    site=COGSGUARD_ARENA,
    num_cogs=4,
    max_steps=1000,
    teams={"cogs": CogTeam(name="cogs", num_agents=4, wealth=3, initial_hearts=0)},
    variants=[EASY, MinerRewardsVariant()],
)
