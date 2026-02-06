"""Aligner tutorial mission configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from cogames.cogs_vs_clips.clip_difficulty import EASY
from cogames.cogs_vs_clips.cog import CogConfig
from cogames.cogs_vs_clips.mission import CvCMission
from cogames.cogs_vs_clips.sites import COGSGUARD_ARENA
from cogames.cogs_vs_clips.team import CogTeam
from cogames.core import CoGameMissionVariant

if TYPE_CHECKING:
    from mettagrid.config.mettagrid_config import MettaGridConfig


class AlignerRewardsVariant(CoGameMissionVariant):
    name: str = "aligner_rewards"
    description: str = "Aligner-focused reward shaping (scout gear, hearts, junction alignment)."

    @override
    def modify_env(self, mission: CvCMission, env: MettaGridConfig) -> None:
        from cogames.cogs_vs_clips.reward_variants import _apply_aligner  # noqa: PLC0415

        for agent_cfg in env.game.agents:
            rewards = dict(agent_cfg.rewards)
            _apply_aligner(rewards)
            agent_cfg.rewards = rewards


AlignerTutorialMission = CvCMission(
    name="aligner_tutorial",
    description="Learn aligner role - collect hearts, and align neutral junctions (no clips).",
    site=COGSGUARD_ARENA,
    num_cogs=4,
    max_steps=1000,
    cog=CogConfig(heart_limit=3),
    teams={"cogs": CogTeam(name="cogs", num_agents=4, wealth=3, initial_hearts=120)},
    variants=[EASY, AlignerRewardsVariant()],
)
