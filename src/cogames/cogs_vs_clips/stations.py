from typing import Optional

from pydantic import Field

from mettagrid.base_config import Config
from mettagrid.config import vibes
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
    ChestConfig,
    GridObjectConfig,
    InventoryConfig,
    WallConfig,
)

resources = [
    "energy",
    "carbon",
    "oxygen",
    "germanium",
    "silicon",
    "heart",
    "decoder",
    "modulator",
    "resonator",
    "scrambler",
]

# CogsGuard constants
GEAR = ["aligner", "scrambler", "miner", "scout"]
ELEMENTS = ["oxygen", "carbon", "germanium", "silicon"]


HEART_COST = {e: 10 for e in ELEMENTS}
COGSGUARD_ALIGN_COST = {"heart": 1}
COGSGUARD_SCRAMBLE_COST = {"heart": 1}

GEAR_COSTS = {
    "aligner": {"carbon": 3, "oxygen": 1, "germanium": 1, "silicon": 1},
    "scrambler": {"carbon": 1, "oxygen": 3, "germanium": 1, "silicon": 1},
    "miner": {"carbon": 1, "oxygen": 1, "germanium": 3, "silicon": 1},
    "scout": {"carbon": 1, "oxygen": 1, "germanium": 1, "silicon": 3},
}

GEAR_SYMBOLS = {
    "aligner": "🔗",
    "scrambler": "🌀",
    "miner": "⛏️",
    "scout": "🔭",
}


def _neg(recipe: dict[str, int]) -> dict[str, int]:
    return {k: -v for k, v in recipe.items()}


class CvCStationConfig(Config):
    def station_cfg(self) -> GridObjectConfig:
        raise NotImplementedError("Subclasses must implement this method")


class CvCWallConfig(CvCStationConfig):
    def station_cfg(self) -> WallConfig:
        return WallConfig(name="wall", render_symbol=vibes.VIBE_BY_NAME["wall"].symbol)


# ==============================================================================
# CogsGuard Station Configs
# ==============================================================================


class SimpleExtractorConfig(CvCStationConfig):
    """Simple resource extractor with inventory that transfers resources to actors."""

    resource: str = Field(description="The resource to extract")
    initial_amount: int = Field(default=100, description="Initial amount of resource in extractor")
    small_amount: int = Field(default=1, description="Amount extracted without mining equipment")
    large_amount: int = Field(default=10, description="Amount extracted with mining equipment")

    def station_cfg(self) -> ChestConfig:
        return ChestConfig(
            name=f"{self.resource}_extractor",
            map_name=f"{self.resource}_extractor",
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


class JunctionConfig(CvCStationConfig):
    """Supply depot that receives element resources via default vibe into collective."""

    map_name: str = Field(description="Map name for this junction")
    team: Optional[str] = Field(default=None, description="Team/collective this junction belongs to")
    aoe_range: int = Field(default=10, description="Range for AOE effects")
    influence_deltas: dict[str, int] = Field(default_factory=lambda: {"influence": 10, "energy": 100, "hp": 100})
    attack_deltas: dict[str, int] = Field(default_factory=lambda: {"hp": -1, "influence": -100})
    elements: list[str] = Field(default_factory=lambda: ELEMENTS)
    align_cost: dict[str, int] = Field(default_factory=lambda: COGSGUARD_ALIGN_COST)
    scramble_cost: dict[str, int] = Field(default_factory=lambda: COGSGUARD_SCRAMBLE_COST)

    def station_cfg(self) -> GridObjectConfig:
        return GridObjectConfig(
            name="junction",
            map_name=self.map_name,
            render_symbol="📦",
            collective=self.team,
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
                "align": Handler(
                    filters=[isNeutral(), actorHas({"aligner": 1, "influence": 1, **self.align_cost})],
                    mutations=[updateActor(_neg(self.align_cost)), alignToActor()],
                ),
                "scramble": Handler(
                    filters=[isEnemy(), actorHas({"scrambler": 1, **self.scramble_cost})],
                    mutations=[removeAlignment(), updateActor(_neg(self.scramble_cost))],
                ),
            },
        )


class HubConfig(JunctionConfig):
    """Main hub with influence AOE effect. A junction without align/scramble handlers."""

    def station_cfg(self) -> GridObjectConfig:
        return GridObjectConfig(
            name="hub",
            map_name=self.map_name,
            render_name="hub",
            render_symbol="📦",
            collective=self.team,
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


class CogsGuardChestConfig(CvCStationConfig):
    """Chest for heart management in CogsGuard."""

    collective: str = Field(default="cogs", description="Collective this chest belongs to")
    heart_cost: dict[str, int] = Field(default_factory=lambda: HEART_COST)

    def station_cfg(self) -> GridObjectConfig:
        return GridObjectConfig(
            name="chest",
            map_name="chest",
            render_symbol="📦",
            collective=self.collective,
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


class GearStationConfig(CvCStationConfig):
    """Gear station that clears all gear and adds the specified gear type."""

    gear_type: str = Field(description="Type of gear this station provides")
    collective: str = Field(default="cogs", description="Collective this station belongs to")
    gear_costs: dict[str, dict[str, int]] = Field(default_factory=lambda: GEAR_COSTS)

    def station_cfg(self) -> GridObjectConfig:
        cost = self.gear_costs.get(self.gear_type, {})
        return GridObjectConfig(
            name=f"{self.gear_type}_station",
            map_name=f"{self.gear_type}_station",
            render_symbol=GEAR_SYMBOLS.get(self.gear_type, "⚙️"),
            collective=self.collective,
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
