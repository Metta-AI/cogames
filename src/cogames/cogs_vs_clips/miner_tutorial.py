"""Miner tutorial mission configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from cogames.cogs_vs_clips.clip_difficulty import EASY
from cogames.cogs_vs_clips.mission import CvCMission
from cogames.cogs_vs_clips.sites import make_cogsguard_machina1_site
from cogames.cogs_vs_clips.team import CogTeam
from cogames.cogs_vs_clips.variants import NoVibesVariant
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
            _apply_miner(rewards, agent_cfg)
            agent_cfg.rewards = rewards


MinerTutorialMission = CvCMission(
    name="miner_tutorial",
    description="Learn miner role - resource extraction and deposits (no clips, no vibes).",
    site=make_cogsguard_machina1_site(4),
    num_cogs=4,
    max_steps=1000,
    teams={"cogs": CogTeam(name="cogs", num_agents=4)},
    variants=[EASY, NoVibesVariant(), MinerRewardsVariant()],
)
