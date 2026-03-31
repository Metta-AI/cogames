"""Energy variant: gives agents an energy pool and movement cost."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from pydantic import Field

from cogames.core import CoGameMissionVariant
from mettagrid.config.mettagrid_config import MettaGridConfig, ResourceLimitsConfig

if TYPE_CHECKING:
    from cogames.games.cogs_vs_clips.missions.mission import CvCMission


class EnergyVariant(CoGameMissionVariant):
    """Add energy inventory, initial energy, and movement cost to agents."""

    name: str = "energy"
    description: str = "Agents have energy that is consumed by movement."

    limit: int = Field(default=20)
    modifiers: dict[str, int] = Field(default_factory=dict)
    initial: int = Field(default=100)
    action_cost: dict[str, int] = Field(default_factory=lambda: {"energy": 4})

    @override
    def modify_env(self, mission: CvCMission, env: MettaGridConfig) -> None:
        env.game.add_resource("energy")

        for agent in env.game.agents:
            agent.inventory.limits.setdefault(
                "energy",
                ResourceLimitsConfig(base=self.limit, resources=["energy"], modifiers=self.modifiers),
            )
            agent.inventory.initial["energy"] = self.initial

        env.game.actions.move.consumed_resources = self.action_cost
