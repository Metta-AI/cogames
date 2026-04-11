"""Damage variant: adds HP resource with passive drain."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from pydantic import Field

from cogames.core import CoGameMissionVariant
from mettagrid.config.filter import isNot
from mettagrid.config.handler_config import Handler, actorHas, allOf, updateActor
from mettagrid.config.mettagrid_config import MettaGridConfig, ResourceLimitsConfig
from mettagrid.config.mutation import ClearInventoryMutation, EntityTarget

if TYPE_CHECKING:
    from cogames.games.cogs_vs_clips.missions.mission import CvCMission


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
            inv.limits["hp"] = ResourceLimitsConfig(base=self.limit, resources=["hp"], modifiers=self.modifiers)
            inv.initial["hp"] = self.initial

            hp_regen = Handler(name="hp_regen", mutations=[updateActor({"hp": self.regen})])
            agent.on_tick = allOf([agent.on_tick, hp_regen])

            if self.destroy_items:
                hp_death = Handler(
                    name="hp_death",
                    filters=[isNot(actorHas({"hp": 1}))],
                    mutations=[
                        ClearInventoryMutation(target=EntityTarget.ACTOR, limit_name=item)
                        for item in self.destroy_items
                    ],
                )
                agent.on_tick = allOf([agent.on_tick, hp_death])
