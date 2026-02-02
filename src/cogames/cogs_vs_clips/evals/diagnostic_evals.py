from __future__ import annotations

from typing import Dict

from pydantic import Field

from cogames.cogs_vs_clips.cog import CogConfig
from cogames.cogs_vs_clips.mission import CvCMission
from cogames.cogs_vs_clips.sites import EVALS, get_map
from cogames.core import CoGameSite
from mettagrid.config.handler_config import Handler
from mettagrid.config.mettagrid_config import (
    MettaGridConfig,
)
from mettagrid.config.mutation.resource_mutation import updateActor

RESOURCE_NAMES: tuple[str, ...] = ("carbon", "oxygen", "germanium", "silicon")


# Generous cog config for diagnostic missions: high limits and full energy regen
_GENEROUS_COG = CogConfig(
    gear_limit=255,
    hp_limit=255,
    heart_limit=255,
    energy_limit=255,
    cargo_limit=255,
    initial_energy=255,
    initial_hp=100,
    energy_regen=255,
    hp_regen=0,
    influence_regen=0,
)


class _DiagnosticMissionBase(CvCMission):
    """Base class for minimal diagnostic evaluation missions."""

    site: CoGameSite = EVALS
    cog: CogConfig = Field(default_factory=lambda: _GENEROUS_COG.model_copy())

    map_name: str = Field(default="evals/diagnostic_eval_template.map")
    max_steps: int = Field(default=250)
    required_agents: int | None = Field(default=None)

    inventory_seed: Dict[str, int] = Field(default_factory=dict)
    # If True, give agents high energy capacity and regen (overridden by specific missions)
    generous_energy: bool = Field(default=True)

    def configure_env(self, cfg: MettaGridConfig) -> None:  # pragma: no cover - hook for subclasses
        """Hook for mission-specific environment alterations."""

    def make_env(self) -> MettaGridConfig:
        """Override make_env to use the mission's map_name instead of site.map_builder."""
        forced_map = get_map(self.map_name)
        # Temporarily override site.map_builder so parent make_env uses the correct map
        original_map_builder = self.site.map_builder
        self.site.map_builder = forced_map
        try:
            cfg = super().make_env()
            # Apply diagnostic-specific modifications
            cfg.game.map_builder = forced_map
            cfg.game.max_steps = self.max_steps
            self._apply_inventory_seed(cfg)
            self.configure_env(cfg)
            return cfg
        finally:
            # Restore original map_builder
            self.site.map_builder = original_map_builder

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _apply_inventory_seed(self, cfg: MettaGridConfig) -> None:
        if not self.inventory_seed:
            return
        seed = dict(cfg.game.agent.inventory.initial)
        seed.update(self.inventory_seed)
        cfg.game.agent.inventory.initial = seed
        # Also apply to per-agent configs (used by CvCMission)
        for agent_cfg in cfg.game.agents:
            agent_seed = dict(agent_cfg.inventory.initial)
            agent_seed.update(self.inventory_seed)
            agent_cfg.inventory.initial = agent_seed


# ----------------------------------------------------------------------
# Diagnostic missions (no assemblers)
# ----------------------------------------------------------------------


# Chest navigation: agents start with a heart and must deposit it
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


# Chest deposit: explicitly single-agent defaults
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
    # Disable generous energy for this eval
    generous_energy: bool = False
    max_steps: int = Field(default=250)

    def configure_env(self, cfg: MettaGridConfig) -> None:
        # Set starting energy to 60 and no regen
        agent = cfg.game.agent
        agent.inventory.initial = dict(agent.inventory.initial)
        agent.inventory.initial["energy"] = 60
        agent.on_tick = {"regen": Handler(mutations=[updateActor({"energy": 0})])}


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
    # Disable generous energy for this eval
    generous_energy: bool = False
    max_steps: int = Field(default=350)

    def configure_env(self, cfg: MettaGridConfig) -> None:
        # Set starting energy to 60 and no regen
        agent = cfg.game.agent
        agent.inventory.initial = dict(agent.inventory.initial)
        agent.inventory.initial["energy"] = 60
        agent.on_tick = {"regen": Handler(mutations=[updateActor({"energy": 0})])}


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
