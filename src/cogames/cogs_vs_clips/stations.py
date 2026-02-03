from typing import Optional

from pydantic import Field

from cogames.cogs_vs_clips.config import CvCConfig
from mettagrid.base_config import Config
from mettagrid.config.handler_config import (
    AOEConfig,
    ClearInventoryMutation,
    EntityTarget,
    Handler,
    actorCollectiveHas,
    actorHas,
    alignToActor,
    collectiveDeposit,
    collectiveWithdraw,
    isAlignedToActor,
    isEnemy,
    isNeutral,
    removeAlignment,
    targetCollectiveHas,
    updateActor,
    updateTarget,
    updateTargetCollective,
    withdraw,
)
from mettagrid.config.mettagrid_config import (
    GridObjectConfig,
    InventoryConfig,
    WallConfig,
)


def _neg(recipe: dict[str, int]) -> dict[str, int]:
    return {k: -v for k, v in recipe.items()}


class CvCStationConfig(Config):
    def station_cfg(self) -> GridObjectConfig:
        raise NotImplementedError("Subclasses must implement this method")


class CvCWallConfig(CvCStationConfig):
    def station_cfg(self) -> WallConfig:
        return WallConfig(name="wall", render_symbol="⬛")


class CvCExtractorConfig(CvCStationConfig):
    """Simple resource extractor with inventory that transfers resources to actors."""

    resource: str = Field(description="The resource to extract")
    initial_amount: int = Field(default=100, description="Initial amount of resource in extractor")
    small_amount: int = Field(default=1, description="Amount extracted without mining equipment")
    large_amount: int = Field(default=10, description="Amount extracted with mining equipment")

    def station_cfg(self) -> GridObjectConfig:
        return GridObjectConfig(
            name=f"{self.resource}_extractor",
            render_symbol="📦",
            on_use_handlers={
                # Order matters: miner first so agents with miner gear get the bonus
                "miner": Handler(
                    filters=[actorHas({"miner": 1})],
                    mutations=[withdraw({self.resource: self.large_amount}, remove_when_empty=True)],
                ),
                "extract": Handler(
                    filters=[],
                    mutations=[withdraw({self.resource: self.small_amount}, remove_when_empty=True)],
                ),
            },
            inventory=InventoryConfig(initial={self.resource: self.initial_amount}),
        )


class CvCJunctionConfig(CvCStationConfig):
    """Supply depot that receives element resources via default vibe into collective."""

    aoe_range: int = Field(default=10, description="Range for AOE effects")
    influence_deltas: dict[str, int] = Field(default_factory=lambda: {"influence": 10, "energy": 100, "hp": 100})
    attack_deltas: dict[str, int] = Field(default_factory=lambda: {"hp": -1, "influence": -100})

    def station_cfg(self, team: Optional[str] = None) -> GridObjectConfig:
        return GridObjectConfig(
            name="junction",
            render_name="junction",
            render_symbol="📦",
            collective=team,
            aoes={
                "influence": AOEConfig(
                    radius=self.aoe_range,
                    filters=[isAlignedToActor()],
                    mutations=[updateTarget(self.influence_deltas)],
                ),
                "attack": AOEConfig(
                    radius=self.aoe_range,
                    filters=[isEnemy()],
                    mutations=[updateTarget(self.attack_deltas)],
                ),
            },
            on_use_handlers={
                "deposit": Handler(
                    filters=[isAlignedToActor()],
                    mutations=[collectiveDeposit({resource: 100 for resource in CvCConfig.ELEMENTS})],
                ),
                "align": Handler(
                    filters=[isNeutral(), actorHas({"aligner": 1, "influence": 1, **CvCConfig.ALIGN_COST})],
                    mutations=[updateActor(_neg(CvCConfig.ALIGN_COST)), alignToActor()],
                ),
                "scramble": Handler(
                    filters=[isEnemy(), actorHas({"scrambler": 1, **CvCConfig.SCRAMBLE_COST})],
                    mutations=[removeAlignment(), updateActor(_neg(CvCConfig.SCRAMBLE_COST))],
                ),
            },
        )


class CvCHubConfig(CvCStationConfig):
    """Hub station that provides AOE influence/attack and accepts deposits."""

    aoe_range: int = Field(default=10, description="Range for AOE effects")
    influence_deltas: dict[str, int] = Field(default_factory=lambda: {"influence": 10, "energy": 100, "hp": 100})
    attack_deltas: dict[str, int] = Field(default_factory=lambda: {"hp": -1, "influence": -100})
    elements: list[str] = Field(default_factory=lambda: CvCConfig.ELEMENTS)

    def station_cfg(self, team: str, collective: str | None = None) -> GridObjectConfig:
        return GridObjectConfig(
            name=f"{team}:hub",
            render_name="hub",
            render_symbol="📦",
            collective=collective or team,
            aoes={
                "influence": AOEConfig(
                    radius=self.aoe_range,
                    filters=[isAlignedToActor()],
                    mutations=[updateTarget(self.influence_deltas)],
                ),
                "attack": AOEConfig(
                    radius=self.aoe_range,
                    filters=[isEnemy()],
                    mutations=[updateTarget(self.attack_deltas)],
                ),
            },
            on_use_handlers={
                "deposit": Handler(
                    filters=[isAlignedToActor()],
                    mutations=[collectiveDeposit({resource: 100 for resource in self.elements})],
                ),
            },
        )


class CvCChestConfig(CvCStationConfig):
    """Chest station for heart management."""

    heart_cost: dict[str, int] = Field(default_factory=lambda: CvCConfig.HEART_COST)

    def station_cfg(self, team: str, collective: str | None = None) -> GridObjectConfig:
        return GridObjectConfig(
            name=f"{team}:chest",
            render_name="chest",
            render_symbol="📦",
            collective=collective or team,
            on_use_handlers={
                "get_heart": Handler(
                    filters=[isAlignedToActor(), targetCollectiveHas({"heart": 1})],
                    mutations=[collectiveWithdraw({"heart": 1})],
                ),
                "make_heart": Handler(
                    filters=[isAlignedToActor(), targetCollectiveHas(self.heart_cost)],
                    mutations=[
                        updateTargetCollective(_neg(self.heart_cost)),
                        updateActor({"heart": 1}),
                    ],
                ),
            },
        )


class CvCGearStationConfig(CvCStationConfig):
    """Gear station that clears all gear and adds the specified gear type."""

    gear_type: str = Field(description="Type of gear this station provides")
    gear_costs: dict[str, dict[str, int]] = Field(default_factory=lambda: CvCConfig.GEAR_COSTS)
    gear_symbols: dict[str, str] = Field(default_factory=lambda: CvCConfig.GEAR_SYMBOLS)

    def station_cfg(self, team: str, collective: str | None = None) -> GridObjectConfig:
        cost = self.gear_costs.get(self.gear_type, {})
        return GridObjectConfig(
            name=f"{team}:{self.gear_type}",
            render_name=f"{self.gear_type}_station",
            render_symbol=self.gear_symbols[self.gear_type],
            collective=collective or team,
            on_use_handlers={
                "keep_gear": Handler(
                    filters=[isAlignedToActor(), actorHas({self.gear_type: 1})],
                    mutations=[],
                ),
                "change_gear": Handler(
                    filters=[isAlignedToActor(), actorCollectiveHas(cost)],
                    mutations=[
                        ClearInventoryMutation(target=EntityTarget.ACTOR, limit_name="gear"),
                        updateTargetCollective(_neg(cost)),
                        updateActor({self.gear_type: 1}),
                    ],
                ),
            },
        )
