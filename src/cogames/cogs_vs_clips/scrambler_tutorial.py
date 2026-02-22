"""Scrambler tutorial mission configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from cogames.cogs_vs_clips.cog import CogConfig, CogTeam
from cogames.cogs_vs_clips.mission import CvCMission
from cogames.cogs_vs_clips.sites import COGSGUARD_ARENA
from cogames.core import CoGameMissionVariant

if TYPE_CHECKING:
    from mettagrid.config.mettagrid_config import MettaGridConfig


class AllJunctionsClipsAligned(CoGameMissionVariant):
    name: str = "all_clips"
    description: str = "All junctions start clips-aligned (except hub), no further clips spread."

    @override
    def modify_mission(self, mission: CvCMission) -> None:
        mission.clips.disabled = False
        mission.clips.initial_clips_start = 1
        mission.clips.initial_clips_spots = 1000
        # Disable further clips spread
        disable_start = mission.max_steps + 1
        mission.clips.scramble_start = disable_start
        mission.clips.scramble_interval = disable_start
        mission.clips.align_start = disable_start
        mission.clips.align_interval = disable_start

    @override
    def modify_env(self, mission: CvCMission, env: MettaGridConfig) -> None:
        from mettagrid.config.filter import hasTagPrefix, isNot  # noqa: PLC0415

        # Only align junctions without a team — skip the hub junction which is already cogs-aligned.
        clip_event = env.game.events["initial_clips"]
        clip_event.filters = [*clip_event.filters, isNot(hasTagPrefix("team"))]


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
    site=COGSGUARD_ARENA,
    num_cogs=4,
    max_steps=1000,
    cog=CogConfig(heart_limit=3),
    teams={"cogs": CogTeam(name="cogs", num_agents=4, wealth=3, initial_hearts=120)},
    variants=[AllJunctionsClipsAligned(), ScramblerRewardsVariant()],
)
