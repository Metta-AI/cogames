"""Vibes variant: adds the change_vibe action and vibe names."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from pydantic import Field

from cogames.core import CoGameMissionVariant
from mettagrid.config.action_config import ChangeVibeActionConfig
from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.config.vibes import Vibe

if TYPE_CHECKING:
    from cogames.games.cogs_vs_clips.missions.mission import CvCMission


class VibesVariant(CoGameMissionVariant):
    """Add vibes and the change_vibe action."""

    name: str = "vibes"
    description: str = "Agents can express vibes via the change_vibe action."
    vibes: list[Vibe] = Field(
        default_factory=lambda: [
            Vibe("😐", "default"),
            Vibe("❤️", "heart"),
            Vibe("⚙️", "gear"),
            Vibe("🌀", "scrambler"),
            Vibe("🔗", "aligner"),
            Vibe("⛏️", "miner"),
            Vibe("🔭", "scout"),
        ]
    )

    @override
    def modify_env(self, mission: CvCMission, env: MettaGridConfig) -> None:
        env.game.vibe_names = [v.name for v in self.vibes]
        env.game.actions.change_vibe = ChangeVibeActionConfig(vibes=self.vibes)
