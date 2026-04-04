"""Heart variant: adds heart resource and inventory limit to agents."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from pydantic import Field

from cogames.core import CoGameMissionVariant, Deps
from cogames.games.cogs_vs_clips.game.damage import DamageVariant
from cogames.variants import ResolvedDeps
from mettagrid.config.mettagrid_config import MettaGridConfig, ResourceLimitsConfig

if TYPE_CHECKING:
    from cogames.games.cogs_vs_clips.missions.mission import CvCMission


class HeartVariant(CoGameMissionVariant):
    """Add heart resource and inventory limit to agents."""

    name: str = "heart"
    description: str = "Agents can carry hearts for role abilities."
    cost: dict[str, int] = Field(default_factory=dict, description="Element costs to craft a heart, set by mission.")
    limit: int = Field(default=10)

    @override
    def dependencies(self) -> Deps:
        return Deps(optional=[DamageVariant])

    @override
    def configure(self, deps: ResolvedDeps) -> None:
        damage = deps.optional(DamageVariant)
        if damage is not None:
            damage.destroy_items.append("heart")

    @override
    def modify_env(self, mission: CvCMission, env: MettaGridConfig) -> None:
        env.game.add_resource("heart")

        for agent in env.game.agents:
            agent.inventory.limits.setdefault(
                "heart",
                ResourceLimitsConfig(base=self.limit, resources=["heart"]),
            )
