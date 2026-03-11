"""Energy variant: agent energy limits, initial energy, and move cost."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from pydantic import Field

from cogames.core import CoGameMissionVariant
from mettagrid.config.mettagrid_config import MettaGridConfig, ResourceLimitsConfig

if TYPE_CHECKING:
    from cogames.core import CoGameMission


class EnergyVariant(CoGameMissionVariant):
    name: str = "energy"

    limit: int = Field(default=20)
    modifiers: dict[str, int] = Field(default_factory=dict)
    initial: int = Field(default=100)
    action_cost: dict[str, int] = Field(default_factory=lambda: {"energy": 4})

    @override
    def modify_env(self, mission: CoGameMission, env: MettaGridConfig) -> None:
        if "energy" not in env.game.resource_names:
            env.game.resource_names.append("energy")
        for agent in env.game.agents:
            agent.inventory.limits.setdefault(
                "energy",
                ResourceLimitsConfig(min=self.limit, resources=["energy"], modifiers=self.modifiers),
            )
            agent.inventory.initial["energy"] = self.initial
        env.game.actions.move.consumed_resources = self.action_cost
