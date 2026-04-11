from __future__ import annotations

from pathlib import Path
from typing import Dict

from pydantic import Field

from cogames.games.cogs_vs_clips.missions.mission import CvCMission
from mettagrid.config.handler_config import Handler
from mettagrid.config.mettagrid_config import (
    MettaGridConfig,
)
from mettagrid.config.mutation.resource_mutation import updateActor
from mettagrid.map_builder.map_builder import MapBuilderConfig
from mettagrid.mapgen.mapgen import MapGen, MapGenConfig

MAPS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "maps"


def get_map(map_name: str) -> MapGenConfig:
    """Load a map builder configuration from the maps directory."""
    normalized = map_name
    if normalized.startswith("evals/"):
        normalized = f"diagnostic_evals/{normalized.split('/', 1)[1]}"
    map_path = MAPS_DIR / normalized
    if not map_path.exists():
        raise FileNotFoundError(f"Map not found: {map_path}")
    return MapGen.Config(
        instance=MapBuilderConfig.from_uri(str(map_path)),
        instances=1,
        fixed_spawn_order=False,
        instance_border_width=0,
    )


EVALS_MAP_BUILDER = get_map("diagnostic_evals/diagnostic_radial.map")

RESOURCE_NAMES: tuple[str, ...] = ("carbon", "oxygen", "germanium", "silicon")


class _DiagnosticMissionBase(CvCMission):
    """Base class for minimal diagnostic evaluation missions."""

    map_builder: MapGenConfig = EVALS_MAP_BUILDER  # type: ignore[assignment]
    min_cogs: int = 1
    max_cogs: int = 8
    map_name: str = Field(default="evals/diagnostic_eval_template.map")
    max_steps: int = Field(default=250)
    required_agents: int | None = Field(default=None)

    inventory_seed: Dict[str, int] = Field(default_factory=dict)
    generous_energy: bool = Field(default=True)

    def configure_env(self, cfg: MettaGridConfig) -> None:  # pragma: no cover - hook for subclasses
        """Hook for mission-specific environment alterations."""

    def make_env(self) -> MettaGridConfig:
        """Override make_env to use the mission's map_name."""
        forced_map = get_map(self.map_name)
        original_map_builder = self.map_builder
        self.map_builder = forced_map
        cfg = super().make_env()
        cfg.game.map_builder = forced_map
        cfg.game.max_steps = self.max_steps
        self._apply_inventory_seed(cfg)
        self.configure_env(cfg)
        self.map_builder = original_map_builder
        return cfg

    def _apply_inventory_seed(self, cfg: MettaGridConfig) -> None:
        if not self.inventory_seed:
            return
        seed = dict(cfg.game.agent.inventory.initial)
        seed.update(self.inventory_seed)
        cfg.game.agent.inventory.initial = seed
        for agent_cfg in cfg.game.agents:
            agent_seed = dict(agent_cfg.inventory.initial)
            agent_seed.update(self.inventory_seed)
            agent_cfg.inventory.initial = agent_seed


# ----------------------------------------------------------------------
# Diagnostic missions
# ----------------------------------------------------------------------


class DiagnosticChestNavigation1(_DiagnosticMissionBase):
    name: str = "diagnostic_chest_navigation1"
    description: str = "Navigate to the chest and deposit a heart."
    map_name: str = "evals/diagnostic_chest_navigation1.map"
    inventory_seed: Dict[str, int] = Field(default_factory=lambda: {"heart": 1})
    max_steps: int = Field(default=250)
    required_agents: int | None = 1


class DiagnosticChestNavigation2(_DiagnosticMissionBase):
    name: str = "diagnostic_chest_navigation2"
    description: str = "Navigate through obstacles to deposit a heart."
    map_name: str = "evals/diagnostic_chest_navigation2.map"
    inventory_seed: Dict[str, int] = Field(default_factory=lambda: {"heart": 1})
    max_steps: int = Field(default=250)
    required_agents: int | None = 1


class DiagnosticChestNavigation3(_DiagnosticMissionBase):
    name: str = "diagnostic_chest_navigation3"
    description: str = "Navigate obstacles to deposit a heart."
    map_name: str = "evals/diagnostic_chest_navigation3.map"
    inventory_seed: Dict[str, int] = Field(default_factory=lambda: {"heart": 1})
    max_steps: int = Field(default=250)
    required_agents: int | None = 1


class DiagnosticChestDepositNear(_DiagnosticMissionBase):
    name: str = "diagnostic_chest_deposit_near"
    description: str = "Deposit a carried heart into a nearby chest."
    map_name: str = "evals/diagnostic_chest_near.map"
    inventory_seed: Dict[str, int] = Field(default_factory=lambda: {"heart": 1})
    required_agents: int | None = 1
    max_steps: int = Field(default=250)


class DiagnosticChestDepositSearch(_DiagnosticMissionBase):
    name: str = "diagnostic_chest_deposit_search"
    description: str = "Find the chest outside the initial FOV and deposit a heart."
    map_name: str = "evals/diagnostic_chest_search.map"
    inventory_seed: Dict[str, int] = Field(default_factory=lambda: {"heart": 1})
    required_agents: int | None = 1
    max_steps: int = Field(default=250)


class DiagnosticChargeUp(_DiagnosticMissionBase):
    name: str = "diagnostic_charge_up"
    description: str = "Agent starts low on energy and must charge to proceed."
    map_name: str = "evals/diagnostic_charge_up.map"
    required_agents: int | None = 1
    inventory_seed: Dict[str, int] = Field(default_factory=lambda: {"heart": 1})
    generous_energy: bool = False
    max_steps: int = Field(default=250)

    def configure_env(self, cfg: MettaGridConfig) -> None:
        agent = cfg.game.agent
        agent.inventory.initial = dict(agent.inventory.initial)
        agent.inventory.initial["energy"] = 60
        agent.on_tick = Handler(name="regen", mutations=[updateActor({"energy": 0})])


class DiagnosticMemory(_DiagnosticMissionBase):
    name: str = "diagnostic_memory"
    description: str = "Harder memory challenge with longer distance to chest."
    map_name: str = "evals/diagnostic_memory.map"
    inventory_seed: Dict[str, int] = Field(default_factory=lambda: {"heart": 1})
    required_agents: int | None = 1
    max_steps: int = Field(default=110)


# ----------------------------------------------------------------------
# Hard versions of diagnostics (same maps, more time)
# ----------------------------------------------------------------------


class DiagnosticChestNavigation1Hard(_DiagnosticMissionBase):
    name: str = "diagnostic_chest_navigation1_hard"
    description: str = "Navigate to the chest and deposit a heart (hard)."
    map_name: str = "evals/diagnostic_chest_navigation1_hard.map"
    inventory_seed: Dict[str, int] = Field(default_factory=lambda: {"heart": 1})
    max_steps: int = Field(default=350)
    required_agents: int | None = 1


class DiagnosticChestNavigation2Hard(_DiagnosticMissionBase):
    name: str = "diagnostic_chest_navigation2_hard"
    description: str = "Navigate through obstacles to deposit a heart (hard)."
    map_name: str = "evals/diagnostic_chest_navigation2_hard.map"
    inventory_seed: Dict[str, int] = Field(default_factory=lambda: {"heart": 1})
    max_steps: int = Field(default=350)
    required_agents: int | None = 1


class DiagnosticChestNavigation3Hard(_DiagnosticMissionBase):
    name: str = "diagnostic_chest_navigation3_hard"
    description: str = "Navigate obstacles to deposit a heart (hard)."
    map_name: str = "evals/diagnostic_chest_navigation3_hard.map"
    inventory_seed: Dict[str, int] = Field(default_factory=lambda: {"heart": 1})
    max_steps: int = Field(default=350)
    required_agents: int | None = 1


class DiagnosticChestDepositSearchHard(_DiagnosticMissionBase):
    name: str = "diagnostic_chest_deposit_search_hard"
    description: str = "Find the chest outside the initial FOV and deposit a heart (hard)."
    map_name: str = "evals/diagnostic_chest_search_hard.map"
    inventory_seed: Dict[str, int] = Field(default_factory=lambda: {"heart": 1})
    required_agents: int | None = 1
    max_steps: int = Field(default=350)


class DiagnosticChargeUpHard(_DiagnosticMissionBase):
    name: str = "diagnostic_charge_up_hard"
    description: str = "Agent starts low on energy and must charge to proceed (hard)."
    map_name: str = "evals/diagnostic_charge_up_hard.map"
    required_agents: int | None = 1
    inventory_seed: Dict[str, int] = Field(default_factory=lambda: {"heart": 1})
    generous_energy: bool = False
    max_steps: int = Field(default=350)

    def configure_env(self, cfg: MettaGridConfig) -> None:
        agent = cfg.game.agent
        agent.inventory.initial = dict(agent.inventory.initial)
        agent.inventory.initial["energy"] = 60
        agent.on_tick = Handler(name="regen", mutations=[updateActor({"energy": 0})])


class DiagnosticMemoryHard(_DiagnosticMissionBase):
    name: str = "diagnostic_memory_hard"
    description: str = "Harder memory challenge with longer distance to chest (hard)."
    map_name: str = "evals/diagnostic_memory_hard.map"
    inventory_seed: Dict[str, int] = Field(default_factory=lambda: {"heart": 1})
    required_agents: int | None = 1
    max_steps: int = Field(default=170)


DIAGNOSTIC_EVALS: list[type[_DiagnosticMissionBase]] = [
    DiagnosticChestNavigation1,
    DiagnosticChestNavigation2,
    DiagnosticChestNavigation3,
    DiagnosticChestDepositNear,
    DiagnosticChestDepositSearch,
    DiagnosticChargeUp,
    DiagnosticMemory,
    # Hard versions
    DiagnosticChestNavigation1Hard,
    DiagnosticChestNavigation2Hard,
    DiagnosticChestNavigation3Hard,
    DiagnosticChestDepositSearchHard,
    DiagnosticChargeUpHard,
    DiagnosticMemoryHard,
]
