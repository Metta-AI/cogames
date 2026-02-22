from __future__ import annotations

from typing import Iterator

from pydantic import Field

from cogames.cogs_vs_clips.config import CvCConfig
from cogames.cogs_vs_clips.team import TeamConfig
from mettagrid.base_config import Config
from mettagrid.config.filter import sharedTagPrefix
from mettagrid.config.game_value import InventoryValue, TagCountValue
from mettagrid.config.handler_config import (
    ClearInventoryMutation,
    EntityTarget,
    Handler,
    actorHas,
    queryDelta,
    updateActor,
)
from mettagrid.config.mettagrid_config import (
    AgentConfig,
    GridObjectConfig,
    InventoryConfig,
    ResourceLimitsConfig,
)
from mettagrid.config.mutation.game_value_mutation import SetGameValueMutation
from mettagrid.config.reward_config import reward


class CogConfig(Config):
    """Configuration for cog agents in CogsGuard game mode."""

    gear_limit: int = Field(default=1)
    hp_limit: int = Field(default=100)
    heart_limit: int = Field(default=10)
    energy_limit: int = Field(default=20)
    cargo_limit: int = Field(default=4)

    hp_modifiers: dict[str, int] = Field(default_factory=lambda: {"scout": 400, "scrambler": 200})
    energy_modifiers: dict[str, int] = Field(default_factory=lambda: {"scout": 100})
    cargo_modifiers: dict[str, int] = Field(default_factory=lambda: {"miner": 40})

    initial_energy: int = Field(default=100)
    initial_hp: int = Field(default=50)
    initial_solar: int = Field(default=1)

    hp_regen: int = Field(default=-1)
    action_cost: dict[str, int] = Field(default_factory=lambda: {"energy": 4})

    def agent_config(self, team: TeamConfig, max_steps: int) -> AgentConfig:
        return AgentConfig(
            team_id=team.team_id,
            tags=[team.team_tag()],
            inventory=InventoryConfig(
                limits={
                    "hp": ResourceLimitsConfig(min=self.hp_limit, resources=["hp"], modifiers=self.hp_modifiers),
                    "gear": ResourceLimitsConfig(max=self.gear_limit, resources=CvCConfig.GEAR, modifiers={"hp": 100}),
                    "heart": ResourceLimitsConfig(max=self.heart_limit, resources=["heart"], modifiers={"hp": 100}),
                    "energy": ResourceLimitsConfig(
                        min=self.energy_limit, resources=["energy"], modifiers=self.energy_modifiers
                    ),
                    "cargo": ResourceLimitsConfig(
                        min=self.cargo_limit, resources=CvCConfig.ELEMENTS, modifiers=self.cargo_modifiers
                    ),
                },
                initial={"energy": self.initial_energy, "hp": self.initial_hp, "solar": self.initial_solar},
            ),
            on_tick={
                "regen": Handler(mutations=[updateActor({"hp": self.hp_regen})]),
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
                    TagCountValue(tag=team.net_tag()),
                    weight=1.0 / max_steps,
                    per_tick=True,
                ),
            },
        )


class CogTeam(TeamConfig):
    """Configuration for a cogs team."""

    name: str = Field(default="cogs", description="Team name used for tags and team identity")
    short_name: str = Field(default="c", description="Short prefix used for map object names")
    wealth: int = Field(default=1, description="Wealth multiplier for initial resources")
    initial_hearts: int | None = Field(default=None, description="Override initial hearts (default: 20 * wealth)")
    num_agents: int = Field(default=8, ge=1, description="Number of agents in the team")

    def gear_station(self, gear_type: str) -> GridObjectConfig:
        cost = CvCConfig.GEAR_COSTS[gear_type]
        hq = self.hub_query()
        return GridObjectConfig(
            name=f"{self.short_name}:{gear_type}",
            render_name=f"{gear_type}_station",
            render_symbol=CvCConfig.GEAR_SYMBOLS[gear_type],
            tags=[f"team:{self.name}"],
            on_use_handlers={
                "keep_gear": Handler(
                    filters=[sharedTagPrefix("team:"), actorHas({gear_type: 1})],
                    mutations=[],
                ),
                "change_gear": Handler(
                    filters=[sharedTagPrefix("team:"), *self.hub_has(cost)],
                    mutations=[
                        ClearInventoryMutation(target=EntityTarget.ACTOR, limit_name="gear"),
                        queryDelta(hq, {k: -v for k, v in cost.items()}),
                        updateActor({gear_type: 1}),
                    ],
                ),
            },
        )

    def _gear_stations(self) -> Iterator[tuple[str, GridObjectConfig]]:
        for gear in CvCConfig.GEAR:
            yield (f"{self.short_name}:{gear}", self.gear_station(gear))

    def _hub_inventory(self) -> InventoryConfig:
        max_element_cost = max(cost for gear in CvCConfig.GEAR_COSTS.values() for cost in gear.values())
        per_element = self.num_agents * max_element_cost * self.wealth
        initial_hearts = (
            self.initial_hearts if self.initial_hearts is not None else CvCConfig.INITIAL_HEARTS * self.wealth
        )
        return InventoryConfig(
            limits={
                "resources": ResourceLimitsConfig(min=10000, resources=CvCConfig.ELEMENTS),
            },
            initial={
                **{element: per_element for element in CvCConfig.ELEMENTS},
                "heart": initial_hearts,
            },
        )
