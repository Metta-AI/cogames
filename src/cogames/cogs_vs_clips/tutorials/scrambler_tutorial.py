"""Scrambler tutorial mission configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from cogames.cogs_vs_clips.cog import CogConfig, CogTeam
from cogames.cogs_vs_clips.mission import CvCMission
from cogames.cogs_vs_clips.sites import COGSGUARD_MACHINA_1
from cogames.core import CoGameMissionVariant

if TYPE_CHECKING:
    from mettagrid.config.mettagrid_config import MettaGridConfig


class OverrunVariant(CoGameMissionVariant):
    name: str = "overrun"
    description: str = "All junctions start clips-aligned (except hub). No further clips spread."

    @override
    def modify_mission(self, mission: CvCMission) -> None:
        mission.clips.disabled = False
        # Fire the align event once at step 1 targeting all junctions.
        mission.clips.align_start = 1
        mission.clips.align_interval = mission.max_steps + 1
        mission.clips.align_all_neutral = True
        mission.clips.align_unlimited_targets = True
        # Disable further clips spread/scramble after initialization.
        disable_start = mission.max_steps + 1
        mission.clips.scramble_start = disable_start
        mission.clips.scramble_interval = disable_start


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
    description="Learn scrambler role - acquire scrambler gear and scramble enemy junctions.",
    site=COGSGUARD_MACHINA_1,
    num_cogs=4,
    max_steps=1000,
    cog=CogConfig(heart_limit=3),
    teams={"cogs": CogTeam(name="cogs", num_agents=4, wealth=3, initial_hearts=120)},
    variants=[OverrunVariant(), ScramblerRewardsVariant()],
)
