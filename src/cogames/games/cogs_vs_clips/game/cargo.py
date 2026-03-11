"""Cargo variant: gives agents inventory space for carrying elements."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from pydantic import Field

from cogames.core import CoGameMissionVariant, Deps
from cogames.games.cogs_vs_clips.game.elements import ElementsVariant
from cogames.variants import ResolvedDeps
from mettagrid.config.mettagrid_config import MettaGridConfig, ResourceLimitsConfig

if TYPE_CHECKING:
    from cogames.cogs_vs_clips.mission import CvCMission


class CargoLimitVariant(CoGameMissionVariant):
    """Add a cargo inventory limit to all agents so they can carry elements."""

    name: str = "cargo_limit"
    description: str = "Agents can carry elements (oxygen, carbon, germanium, silicon)."

    limit: int = Field(default=4)
    modifiers: dict[str, int] = Field(default_factory=dict)

    @override
    def dependencies(self) -> Deps:
        return Deps(required=[ElementsVariant])

    @override
    def configure(self, deps: ResolvedDeps) -> None:
        deps.required(ElementsVariant)

    @override
    def modify_env(self, mission: CvCMission, env: MettaGridConfig) -> None:
        elements = mission.required_variant(ElementsVariant).elements
        for agent in env.game.agents:
            agent.inventory.limits.setdefault(
                "cargo",
                ResourceLimitsConfig(min=self.limit, resources=elements, modifiers=self.modifiers),
            )
