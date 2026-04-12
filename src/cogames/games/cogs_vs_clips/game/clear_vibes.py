"""Clear vibes variant: heart vibe clears on any bump."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from cogames.core import CoGameMissionVariant, Deps
from cogames.games.cogs_vs_clips.game.vibes import VibesVariant
from mettagrid.config.filter import targetVibe
from mettagrid.config.handler_config import Handler
from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.config.mutation import changeTargetVibe

if TYPE_CHECKING:
    from cogames.games.cogs_vs_clips.missions.mission import CvCMission


class ClearVibesVariant(CoGameMissionVariant):
    """Bumping a heart-vibed object clears its vibe."""

    name: str = "clear_vibes"
    description: str = "Heart vibe clears on any bump."

    @override
    def dependencies(self) -> Deps:
        return Deps(required=[VibesVariant])

    @override
    def modify_env(self, mission: CvCMission, env: MettaGridConfig) -> None:
        clear_handler = Handler(
            filters=[targetVibe("heart")],
            mutations=[changeTargetVibe("default")],
        )
        for agent in env.game.agents:
            agent.on_after_use_handler = clear_handler
