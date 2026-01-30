"""
Difficulty Variants for CoGames Missions

This module defines difficulty levels that can be applied to any mission to create
varied challenges. Each difficulty level modifies mission-level parameters like
energy regen, move cost, and capacity limits.
"""

from __future__ import annotations

import logging
from typing import override

from pydantic import Field

from cogames.cogs_vs_clips.mission import Mission, MissionVariant
from mettagrid.config.mettagrid_config import MettaGridConfig

logger = logging.getLogger(__name__)


# Allow zero to persist for difficulties that force no passive regen
ENERGY_REGEN_FLOOR = 0


# =============================================================================
# Difficulty Registry
# =============================================================================


class DifficultyLevel(MissionVariant):
    """Configuration for a difficulty level."""

    name: str = Field(description="Difficulty name (easy, medium, hard, brutal, etc.)")
    description: str = Field(description="What makes this difficulty challenging", default="")

    # Energy regen multiplier (relative to mission baseline)
    energy_regen_mult: float = Field(default=1.0)

    # Absolute overrides (if set, ignore multipliers)
    energy_regen_override: int | None = Field(default=None)
    move_energy_cost_override: int | None = Field(default=None)
    energy_capacity_override: int | None = Field(default=None)
    cargo_capacity_override: int | None = Field(default=None)
    max_steps_override: int | None = Field(default=None)

    @override
    def modify_mission(self, mission: Mission):
        """Apply a difficulty level to a mission instance."""
        # Energy regen
        if self.energy_regen_override is not None:
            mission.cog.energy_regen = self.energy_regen_override
        else:
            mission.cog.energy_regen = max(0, int(mission.cog.energy_regen * self.energy_regen_mult))

        # Mission-level overrides
        if self.move_energy_cost_override is not None:
            mission.cog.move_energy_cost = self.move_energy_cost_override
        if self.energy_capacity_override is not None:
            mission.cog.energy_limit = self.energy_capacity_override
        if self.cargo_capacity_override is not None:
            mission.cog.cargo_limit = self.cargo_capacity_override

    @override
    def modify_env(self, mission: Mission, env: MettaGridConfig):
        if self.max_steps_override is not None:
            env.game.max_steps = self.max_steps_override


# =============================================================================
# Standard Difficulty Levels
# =============================================================================

STANDARD = DifficultyLevel(
    name="standard",
    description="Baseline mission parameters (legacy medium)",
)

HARD = DifficultyLevel(
    name="hard",
    description="Minimal passive regen and higher move cost",
    energy_regen_override=1,  # Minimal regen prevents deadlock
    move_energy_cost_override=2,
)

SINGLE_USE = DifficultyLevel(
    name="single_use",
    description="Minimal regen - no second chances",
    energy_regen_override=1,
)

SPEED_RUN = DifficultyLevel(
    name="speed_run",
    description="Short clock, cheap movement",
    energy_regen_override=2,
    move_energy_cost_override=1,
    max_steps_override=600,
)

ENERGY_CRISIS = DifficultyLevel(
    name="energy_crisis",
    description="Minimal passive regen - plan every move",
    energy_regen_override=1,  # Minimal regen prevents deadlock
)

# Export variants for use with --variant CLI flag.
# Ordered in canonical difficulty order.
DIFFICULTY_VARIANTS: list[DifficultyLevel] = [
    STANDARD,
    HARD,
    SINGLE_USE,
    SPEED_RUN,
    ENERGY_CRISIS,
]


def get_difficulty(name: str) -> DifficultyLevel:
    """Get a difficulty level by name."""
    return next(difficulty for difficulty in DIFFICULTY_VARIANTS if difficulty.name == name)


def list_difficulties() -> None:
    """Print all available difficulty levels."""
    print("\nAvailable Difficulty Levels")
    print("=" * 80)
    for diff in DIFFICULTY_VARIANTS:
        print(f"\n{diff.name.upper()}: {diff.description}")
        print(f"  Energy regen mult: {diff.energy_regen_mult}")


if __name__ == "__main__":
    list_difficulties()
