"""Elements variant: adds element resources to the game."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from pydantic import Field

from cogames.core import CoGameMissionVariant
from mettagrid.config.mettagrid_config import MettaGridConfig

if TYPE_CHECKING:
    from cogames.games.cogs_vs_clips.missions.mission import CvCMission


class ElementsVariant(CoGameMissionVariant):
    """Register element resources in the game config."""

    name: str = "elements"
    description: str = "Add element resources (oxygen, carbon, germanium, silicon)."
    elements: list[str] = Field(default_factory=lambda: ["oxygen", "carbon", "germanium", "silicon"])

    @override
    def modify_env(self, mission: CvCMission, env: MettaGridConfig) -> None:
        for element in self.elements:
            env.game.add_resource(element)
