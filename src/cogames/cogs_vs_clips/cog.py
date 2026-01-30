from __future__ import annotations

from pydantic import Field

from mettagrid.base_config import Config
from mettagrid.config.handler_config import Handler
from mettagrid.config.mettagrid_config import (
    AgentConfig,
    InventoryConfig,
    ResourceLimitsConfig,
)
from mettagrid.config.mutation.resource_mutation import updateActor


class CogConfig(Config):
    """Configuration for cog agents in CogsGuard game mode."""

    # Inventory limits
    gear_limit: int = Field(default=1)
    hp_limit: int = Field(default=100)
    heart_limit: int = Field(default=10)
    energy_limit: int = Field(default=20)
    cargo_limit: int = Field(default=4)
    influence_limit: int = Field(default=0)

    # Inventory modifiers by gear type
    hp_modifiers: dict[str, int] = Field(default_factory=lambda: {"scout": 400, "scrambler": 200})
    energy_modifiers: dict[str, int] = Field(default_factory=lambda: {"scout": 100})
    cargo_modifiers: dict[str, int] = Field(default_factory=lambda: {"miner": 40})
    influence_modifiers: dict[str, int] = Field(default_factory=lambda: {"aligner": 20})

    # Initial inventory
    initial_energy: int = Field(default=100)
    initial_hp: int = Field(default=50)

    # Regen amounts
    energy_regen: int = Field(default=1)
    hp_regen: int = Field(default=-1)
    influence_regen: int = Field(default=-1)

    # Movement cost
    move_energy_cost: int = Field(default=3)

    def agent_config(self, gear: list[str], elements: list[str]) -> AgentConfig:
        """Create an AgentConfig for this cog configuration."""
        return AgentConfig(
            collective="cogs",
            inventory=InventoryConfig(
                limits={
                    "hp": ResourceLimitsConfig(min=self.hp_limit, resources=["hp"], modifiers=self.hp_modifiers),
                    # when hp == 0, the cog can't hold gear or hearts
                    "gear": ResourceLimitsConfig(max=self.gear_limit, resources=gear, modifiers={"hp": 100}),
                    "heart": ResourceLimitsConfig(max=self.heart_limit, resources=["heart"], modifiers={"hp": 100}),
                    "energy": ResourceLimitsConfig(
                        min=self.energy_limit, resources=["energy"], modifiers=self.energy_modifiers
                    ),
                    "cargo": ResourceLimitsConfig(
                        min=self.cargo_limit, resources=elements, modifiers=self.cargo_modifiers
                    ),
                    "influence": ResourceLimitsConfig(
                        min=self.influence_limit, resources=["influence"], modifiers=self.influence_modifiers
                    ),
                },
                initial={"energy": self.initial_energy, "hp": self.initial_hp},
            ),
            on_tick={
                "regen": Handler(
                    mutations=[
                        updateActor(
                            {
                                "energy": self.energy_regen,
                                "hp": self.hp_regen,
                                "influence": self.influence_regen,
                            }
                        )
                    ]
                )
            },
        )
