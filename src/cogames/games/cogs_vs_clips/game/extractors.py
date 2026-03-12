from __future__ import annotations

from typing import TYPE_CHECKING, override

from pydantic import Field

from cogames.core import CoGameMissionVariant, CvCStationConfig, Deps
from cogames.games.cogs_vs_clips.game.elements import ElementsVariant
from cogames.games.cogs_vs_clips.game.terrain import BuildingsVariant
from cogames.games.cogs_vs_clips.missions.terrain import find_machina_arena
from cogames.variants import ResolvedDeps
from mettagrid.base_config import Config
from mettagrid.config.handler_config import (
    Handler,
    actorHas,
    updateActor,
    withdraw,
)
from mettagrid.config.mettagrid_config import (
    GridObjectConfig,
    InventoryConfig,
    MettaGridConfig,
)

if TYPE_CHECKING:
    from cogames.games.cogs_vs_clips.missions.mission import CvCMission


class CvCExtractorConfig(CvCStationConfig):
    """Station config for a single-resource extractor."""

    resource: str
    initial_amount: int = 200
    small_amount: int = 1

    def station_cfg(self) -> GridObjectConfig:
        return GridObjectConfig(
            name=f"{self.resource}_extractor",
            on_use_handlers={
                "extract": Handler(
                    mutations=[withdraw({self.resource: self.small_amount}, remove_when_empty=True)],
                ),
            },
            inventory=InventoryConfig(initial={self.resource: self.initial_amount}),
        )


class ExtractionHandlerConfig(Config):
    """A specialized extraction handler added before the default extract handler."""

    name: str
    required_resources: dict[str, int] = Field(default_factory=dict)
    cost: dict[str, int] = Field(default_factory=dict)
    amount: int


class ExtractorsVariant(CoGameMissionVariant):
    """Add resource extractors to the environment."""

    name: str = "extractors"
    description: str = "Place extractors for each element on the map."
    extractor_density: float = 0.3
    extraction_handlers: list[ExtractionHandlerConfig] = Field(default_factory=list)
    initial_amount: int = 200

    def add_extraction_handler(
        self, name: str, required_resources: dict[str, int], cost: dict[str, int], amount: int
    ) -> None:
        # Place at the front to give priority to the new handler. This is somewhat fragile.
        self.extraction_handlers.insert(
            0, ExtractionHandlerConfig(name=name, required_resources=required_resources, cost=cost, amount=amount)
        )

    @override
    def dependencies(self) -> Deps:
        return Deps(required=[BuildingsVariant, ElementsVariant])

    @override
    def configure(self, deps: ResolvedDeps) -> None:
        terrain = deps.required(BuildingsVariant)
        for element in deps.required(ElementsVariant).elements:
            terrain.building_density[f"{element}_extractor"] = self.extractor_density
        self.add_extraction_handler("extract", {}, {}, 1)

    @override
    def modify_env(self, mission: CvCMission, env: MettaGridConfig) -> None:
        arena = find_machina_arena(env.game.map_builder)
        if arena is not None:
            arena.hub.corner_bundle = "extractors"

        for resource in mission.required_variant(ElementsVariant).elements:
            key = f"{resource}_extractor"
            env.game.objects[key] = GridObjectConfig(
                name=f"{resource}_extractor",
                on_use_handlers={
                    **{
                        eh.name: Handler(
                            filters=[actorHas({k: v}) for k, v in eh.required_resources.items()],
                            mutations=[
                                updateActor({k: -v for k, v in eh.cost.items()}),
                                withdraw({resource: eh.amount}, remove_when_empty=True),
                            ],
                        )
                        for eh in self.extraction_handlers
                    },
                },
                inventory=InventoryConfig(initial={resource: self.initial_amount}),
            )
