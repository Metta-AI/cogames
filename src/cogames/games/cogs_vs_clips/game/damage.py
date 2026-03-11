"""Damage variant: adds HP resource with passive drain."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from pydantic import Field

from cogames.core import CoGameMissionVariant
from mettagrid.config.filter import isNot
from mettagrid.config.handler_config import Handler, actorHas, updateActor
from mettagrid.config.mettagrid_config import MettaGridConfig, ResourceLimitsConfig
from mettagrid.config.mutation import ClearInventoryMutation, EntityTarget

if TYPE_CHECKING:
    from cogames.cogs_vs_clips.mission import CvCMission


class DamageVariant(CoGameMissionVariant):
    """Add HP resource with passive drain.

    Agents start with initial HP that drains each tick.
    When HP reaches 0, items in destroy_items are cleared.
    """

    name: str = "damage"
    description: str = "HP resource with passive drain and item destruction on death."

    limit: int = Field(default=100)
    modifiers: dict[str, int] = Field(default_factory=dict)
    initial: int = Field(default=50)
    regen: int = Field(default=-1)
    destroy_items: list[str] = Field(default_factory=list)

    @override
    def modify_env(self, mission: CvCMission, env: MettaGridConfig) -> None:
        env.game.add_resource("hp")

        for agent in env.game.agents:
            inv = agent.inventory
            inv.limits["hp"] = ResourceLimitsConfig(min=self.limit, resources=["hp"], modifiers=self.modifiers)
            inv.initial["hp"] = self.initial
            agent.on_tick["hp_regen"] = Handler(mutations=[updateActor({"hp": self.regen})])

            if self.destroy_items:
                agent.on_tick["hp_death"] = Handler(
                    filters=[isNot(actorHas({"hp": 1}))],
                    mutations=[
                        ClearInventoryMutation(target=EntityTarget.ACTOR, limit_name=item)
                        for item in self.destroy_items
                    ],
                )
