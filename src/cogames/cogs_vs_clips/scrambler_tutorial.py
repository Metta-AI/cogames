"""Scrambler tutorial mission configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from cogames.cogs_vs_clips.clip_difficulty import EASY
from cogames.cogs_vs_clips.mission import CvCMission
from cogames.cogs_vs_clips.sites import COGSGUARD_ARENA
from cogames.cogs_vs_clips.team import CogTeam
from cogames.core import CoGameMissionVariant

if TYPE_CHECKING:
    from mettagrid.config.mettagrid_config import MettaGridConfig


class ScramblerRewardsVariant(CoGameMissionVariant):
    name: str = "scrambler_rewards"
    description: str = "Scrambler-focused reward shaping (gear, junction scrambling)."

    @override
    def modify_env(self, mission: CvCMission, env: MettaGridConfig) -> None:
        from cogames.cogs_vs_clips.reward_variants import _apply_scrambler  # noqa: PLC0415

        for agent_cfg in env.game.agents:
            rewards = dict(agent_cfg.rewards)
            _apply_scrambler(rewards)
            agent_cfg.rewards = rewards


ScramblerTutorialMission = CvCMission(
    name="scrambler_tutorial",
    description="Learn scrambler role - scramble enemy junctions (no clips).",
    site=COGSGUARD_ARENA,
    num_cogs=4,
    max_steps=1000,
    teams={"cogs": CogTeam(name="cogs", num_agents=4, wealth=3, initial_hearts=120)},
    variants=[EASY, ScramblerRewardsVariant()],
)
