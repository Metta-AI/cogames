"""Buildings variant: configures building placement on the map."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from pydantic import Field

from cogames.core import CoGameMissionVariant, Deps
from cogames.games.cogs_vs_clips.missions.terrain import find_machina_arena
from cogames.variants import ResolvedDeps
from mettagrid.config.mettagrid_config import MettaGridConfig

if TYPE_CHECKING:
    from cogames.cogs_vs_clips.mission import CvCMission
    from mettagrid.map_builder.map_builder import AnyMapBuilderConfig


class BuildingsVariant(CoGameMissionVariant):
    """Configure which buildings to place on the map and their weights."""

    name: str = "buildings"
    description: str = "Configure building placement on the map."
    building_density: dict[str, float] = Field(
        default_factory=dict, description="Building name -> weight, configured by other variants."
    )

    @override
    def dependencies(self) -> Deps:
        return Deps()

    @override
    def configure(self, deps: ResolvedDeps) -> None:
        pass

    @override
    def modify_env(self, mission: CvCMission, env: MettaGridConfig) -> None:
        if self.building_density:
            _apply_buildings(env.game.map_builder, self.building_density)


def _apply_buildings(builder: AnyMapBuilderConfig, buildings: dict[str, float]) -> None:
    """Apply building weights to a MapGen-based map builder."""
    arena = find_machina_arena(builder)
    if arena is None:
        return
    weights = dict(arena.building_weights or {})
    for name, weight in buildings.items():
        weights.setdefault(name, weight)
    arena.building_weights = weights
