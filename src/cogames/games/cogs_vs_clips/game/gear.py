"""Gear variant: gives agents a gear inventory slot."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from pydantic import Field

from cogames.core import CoGameMissionVariant, Deps
from cogames.games.cogs_vs_clips.game.damage import DamageVariant
from cogames.variants import ResolvedDeps
from mettagrid.config.mettagrid_config import MettaGridConfig, ResourceLimitsConfig

if TYPE_CHECKING:
    from cogames.games.cogs_vs_clips.missions.mission import CvCMission


class GearVariant(CoGameMissionVariant):
    """Add gear inventory limit and register gear item resources."""

    name: str = "gear"
    description: str = "Agents can equip one gear item."
    items: list[str] = Field(default_factory=list, description="Gear item names, registered by role variants.")
    limit: int = Field(default=1)
    station_costs: dict[str, dict[str, int]] = Field(default_factory=dict, description="Station costs by item name.")
    station_symbols: dict[str, str] = Field(default_factory=dict, description="Render symbols by gear item name.")

    destroy_gear_on_death: bool = True

    @override
    def dependencies(self) -> Deps:
        return Deps(optional=[DamageVariant])

    @override
    def configure(self, deps: ResolvedDeps) -> None:
        if self.destroy_gear_on_death:
            d = deps.optional(DamageVariant)
            if d is not None:
                d.destroy_items.append("gear")

    @override
    def modify_env(self, mission: CvCMission, env: MettaGridConfig) -> None:
        for item in self.items:
            env.game.add_resource(item)

        for agent in env.game.agents:
            agent.inventory.limits.setdefault(
                "gear",
                ResourceLimitsConfig(min=self.limit, resources=list(self.items)),
            )
