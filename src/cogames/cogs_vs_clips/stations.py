from __future__ import annotations

from pydantic import Field

from mettagrid.base_config import Config
from mettagrid.config.handler_config import (
    Handler,
    actorHas,
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
    initial_amount: int = Field(default=200, description="Initial amount of resource in extractor")
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
