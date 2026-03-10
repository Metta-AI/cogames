"""Energy variant: agent energy limits, initial energy, and move cost."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from cogames.core import CoGameMissionVariant
from mettagrid.config.mettagrid_config import ResourceLimitsConfig

if TYPE_CHECKING:
    from cogames.cogs_vs_clips.mission import CvCMission
    from mettagrid.config.mettagrid_config import MettaGridConfig


class EnergyVariant(CoGameMissionVariant):
    name: str = "energy"

    @override
    def modify_env(self, mission: CvCMission, env: MettaGridConfig) -> None:
        cog = mission.cog
        for agent in env.game.agents:
            agent.inventory.limits["energy"] = ResourceLimitsConfig(
                min=cog.energy_limit, resources=["energy"], modifiers=cog.energy_modifiers
            )
            agent.inventory.initial["energy"] = cog.initial_energy
        env.game.actions.move.consumed_resources = cog.action_cost
