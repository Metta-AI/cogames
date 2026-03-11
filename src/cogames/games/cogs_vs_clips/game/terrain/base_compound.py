"""BaseCompound variant: configures the base compound (hub) scene with specific placements."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from cogames.games.cogs_vs_clips.missions.terrain import (
    EnvNodeVariant,
    MachinaArena,
    RandomTransform,
)
from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.mapgen.mapgen import MapGen
from mettagrid.mapgen.scenes.compound import Compound, CompoundConfig

if TYPE_CHECKING:
    from cogames.cogs_vs_clips.mission import CvCMission


class BaseCompoundVariant(EnvNodeVariant[CompoundConfig]):
    """Configure the base compound scene layout (hub area, corners, spawns)."""

    @override
    def compat(self, mission: CvCMission) -> bool:
        env = mission.make_env()
        if not isinstance(env.game.map_builder, MapGen.Config):
            return False
        instance = env.game.map_builder.instance
        if not isinstance(instance, Compound.Config):
            return False
        if isinstance(instance, RandomTransform.Config) and isinstance(instance.scene, Compound.Config):
            return True
        if isinstance(instance, MachinaArena.Config):
            return True
        return False

    @classmethod
    def extract_node(cls, env: MettaGridConfig) -> CompoundConfig:
        assert isinstance(env.game.map_builder, MapGen.Config)
        instance = env.game.map_builder.instance

        if isinstance(instance, RandomTransform.Config) and isinstance(instance.scene, Compound.Config):
            return instance.scene

        elif isinstance(instance, MachinaArena.Config):
            return instance.hub

        raise TypeError("BaseCompoundVariant can only be applied to RandomTransform/Compound or MachinaArena scenes")
