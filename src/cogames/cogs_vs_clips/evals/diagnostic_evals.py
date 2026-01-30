from __future__ import annotations

from pathlib import Path
from typing import Dict

from pydantic import Field

from cogames.cogs_vs_clips.cog import CogConfig
from cogames.cogs_vs_clips.mission import Mission, Site
from mettagrid.config.game_value import stat
from mettagrid.config.handler_config import Handler
from mettagrid.config.mettagrid_config import (
    ChestConfig,
    MettaGridConfig,
    ResourceLimitsConfig,
)
from mettagrid.config.mutation.resource_mutation import updateActor
from mettagrid.config.reward_config import reward
from mettagrid.map_builder.map_builder import MapBuilderConfig
from mettagrid.mapgen.mapgen import MapGen

RESOURCE_NAMES: tuple[str, ...] = ("carbon", "oxygen", "germanium", "silicon")

MAPS_DIR = Path(__file__).resolve().parent.parent.parent / "maps"


def get_map(map_name: str) -> MapBuilderConfig:
    """Load a map builder configuration from the local diagnostics directory."""
    normalized = map_name
    if normalized.startswith("evals/"):
        normalized = f"diagnostic_evals/{normalized.split('/', 1)[1]}"
    map_path = MAPS_DIR / normalized
    if not map_path.exists():
        raise FileNotFoundError(f"Diagnostic map not found: {map_path}")
    # Wrap AsciiMapBuilderConfig in MapGen.Config to match standard get_map() behavior
    return MapGen.Config(
        instance=MapBuilderConfig.from_uri(str(map_path)),
        instances=1,  # Force single instance - use spawn points from ASCII map directly
        fixed_spawn_order=False,
        instance_border_width=0,  # Don't add border - maps already have borders built in
    )


EVALS = Site(
    name="evals",
    description="Diagnostic evaluation arenas.",
    map_builder=get_map("evals/diagnostic_radial.map"),
    min_cogs=1,
    max_cogs=4,
)


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

# Same but without generous energy regen (for charge-up diagnostics)
_MODEST_COG = CogConfig(
    gear_limit=255,
    hp_limit=255,
    heart_limit=255,
    energy_limit=255,
    cargo_limit=255,
    initial_energy=255,
    initial_hp=100,
    energy_regen=1,
    hp_regen=0,
    influence_regen=0,
)


class _DiagnosticMissionBase(Mission):
    """Base class for minimal diagnostic evaluation missions."""

    site: Site = EVALS
    cog: CogConfig = Field(default_factory=lambda: _GENEROUS_COG.model_copy())

    map_name: str = Field(default="evals/diagnostic_eval_template.map")
    max_steps: int = Field(default=250)
    required_agents: int | None = Field(default=None)

    inventory_seed: Dict[str, int] = Field(default_factory=dict)
    communal_chest_hearts: int | None = Field(default=None)
    resource_chest_stock: Dict[str, int] = Field(default_factory=dict)
    # If True, give agents high energy capacity and regen (overridden by specific missions)
    generous_energy: bool = Field(default=True)

    # Disable clips events for diagnostic evals
    clips_scramble_start: int = Field(default=99999)
    clips_align_start: int = Field(default=99999)

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
            self._apply_communal_chest(cfg)
            self._apply_resource_chests(cfg)
            # Finally, normalize rewards so a single deposited heart yields at most 1 reward.
            self._apply_heart_reward_cap(cfg)
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

    def _apply_communal_chest(self, cfg: MettaGridConfig) -> None:
        if self.communal_chest_hearts is None:
            return
        chest = cfg.game.objects.get("communal_chest")
        if isinstance(chest, ChestConfig):
            chest.inventory.initial = {"heart": self.communal_chest_hearts}

    def _apply_resource_chests(self, cfg: MettaGridConfig) -> None:
        if not self.resource_chest_stock:
            return
        for resource, amount in self.resource_chest_stock.items():
            chest_cfg = cfg.game.objects.get(f"chest_{resource}")
            if isinstance(chest_cfg, ChestConfig):
                chest_cfg.inventory.initial = {resource: amount}

    def _apply_heart_reward_cap(self, cfg: MettaGridConfig) -> None:
        """Normalize diagnostics so a single deposited heart yields at most 1 reward per episode.

        - Make each agent-deposited heart worth exactly 1.0 reward (credited only to the depositor).
        - Ensure all chests can store at most 1 heart so total reward per episode cannot exceed 1.
        """
        agent_cfg = cfg.game.agent
        rewards = dict(agent_cfg.rewards)
        rewards["chest_heart_deposited_by_agent"] = reward(stat("chest.heart.deposited_by_agent"))
        agent_cfg.rewards = rewards

        # Cap heart capacity for every chest used in diagnostics (communal or resource-specific).
        for _name, obj in cfg.game.objects.items():
            if not isinstance(obj, ChestConfig):
                continue
            # Find existing heart limit or create new one
            heart_limit = obj.inventory.limits.get("heart", ResourceLimitsConfig(min=1, resources=["heart"]))
            heart_limit.min = 1
            obj.inventory.limits["heart"] = heart_limit


# ----------------------------------------------------------------------
# Diagnostics (non-hub)
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
        # Set starting energy to 30 and no regen
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
        # Set starting energy to 30 and no regen
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
