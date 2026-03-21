"""Scout tutorial mission configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from cogames.cogs_vs_clips.clip_difficulty import EASY
from cogames.cogs_vs_clips.cog import CogTeam
from cogames.cogs_vs_clips.mission import CvCMission
from cogames.cogs_vs_clips.sites import COGSGUARD_ARENA
from cogames.core import CoGameMissionVariant

if TYPE_CHECKING:
    from mettagrid.config.mettagrid_config import MettaGridConfig


class ScoutRewardsVariant(CoGameMissionVariant):
    name: str = "scout_rewards"
    description: str = "Scout-focused reward shaping (gear, exploration)."

    @override
    def modify_env(self, mission: CvCMission, env: MettaGridConfig) -> None:
        from cogames.cogs_vs_clips.reward_variants import _apply_scout  # noqa: PLC0415

        for agent_cfg in env.game.agents:
            rewards = dict(agent_cfg.rewards)
            _apply_scout(rewards)
            agent_cfg.rewards = rewards


ScoutTutorialMission = CvCMission(
    name="scout_tutorial",
    description="Learn scout role - exploration and visiting stale cells (no clips).",
    site=COGSGUARD_ARENA,
    num_cogs=4,
    max_steps=1000,
    teams={"cogs": CogTeam(name="cogs", num_agents=4, wealth=3, initial_hearts=0)},
    variants=[EASY, ScoutRewardsVariant()],
)
