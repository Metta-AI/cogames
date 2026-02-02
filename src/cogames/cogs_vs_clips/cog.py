from __future__ import annotations

from pydantic import Field

from cogames.cogs_vs_clips.config import CvCConfig
from mettagrid.base_config import Config
from mettagrid.config.game_value import InventoryValue
from mettagrid.config.game_value import stat as game_stat
from mettagrid.config.handler_config import Handler
from mettagrid.config.mettagrid_config import (
    AgentConfig,
    InventoryConfig,
    ResourceLimitsConfig,
)
from mettagrid.config.mutation.game_value_mutation import SetGameValueMutation
from mettagrid.config.mutation.mutation import EntityTarget
from mettagrid.config.mutation.resource_mutation import updateActor
from mettagrid.config.reward_config import reward


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
    initial_solar: int = Field(default=1)

    # Regen amounts
    hp_regen: int = Field(default=-1)
    influence_regen: int = Field(default=-1)
    action_cost: dict[str, int] = Field(default_factory=lambda: {"energy": 4})

    def agent_config(self, team: str, max_steps: int) -> AgentConfig:
        """Create an AgentConfig for this cog configuration."""
        return AgentConfig(
            collective=team,
            inventory=InventoryConfig(
                limits={
                    "hp": ResourceLimitsConfig(min=self.hp_limit, resources=["hp"], modifiers=self.hp_modifiers),
                    # when hp == 0, the cog can't hold gear or hearts
                    "gear": ResourceLimitsConfig(max=self.gear_limit, resources=CvCConfig.GEAR, modifiers={"hp": 100}),
                    "heart": ResourceLimitsConfig(max=self.heart_limit, resources=["heart"], modifiers={"hp": 100}),
                    "energy": ResourceLimitsConfig(
                        min=self.energy_limit, resources=["energy"], modifiers=self.energy_modifiers
                    ),
                    "cargo": ResourceLimitsConfig(
                        min=self.cargo_limit, resources=CvCConfig.ELEMENTS, modifiers=self.cargo_modifiers
                    ),
                    "influence": ResourceLimitsConfig(
                        min=self.influence_limit, resources=["influence"], modifiers=self.influence_modifiers
                    ),
                },
                initial={"energy": self.initial_energy, "hp": self.initial_hp, "solar": self.initial_solar},
            ),
            on_tick={
                "regen": Handler(
                    mutations=[
                        updateActor(
                            {
                                "hp": self.hp_regen,
                                "influence": self.influence_regen,
                            }
                        )
                    ]
                ),
                "solar_to_energy": Handler(
                    mutations=[
                        SetGameValueMutation(
                            value=InventoryValue(item="energy"),
                            source=InventoryValue(item="solar"),
                            target=EntityTarget.ACTOR,
                        )
                    ]
                ),
            },
            rewards={
                "aligned_junction_held": reward(
                    game_stat("collective.aligned.junction.held"),
                    weight=1.0 / max_steps,
                ),
            },
        )
