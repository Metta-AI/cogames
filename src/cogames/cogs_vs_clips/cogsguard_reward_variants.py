"""Reward preset wiring for the CogsGuard (Cogs vs Clips) mission.

The mission has a single "true" objective signal, plus optional shaping variants.
Reward variants are stackable; each one adds additional shaping signals on top of the
mission's default objective rewards.
"""

from __future__ import annotations

from typing import Literal, Sequence, cast

from mettagrid.config.mettagrid_config import AgentRewards, MettaGridConfig
from mettagrid.mapgen.mapgen import MapGenConfig

CogsGuardRewardVariant = Literal["credit", "milestones", "no_objective", "objective"]

AVAILABLE_REWARD_VARIANTS: tuple[CogsGuardRewardVariant, ...] = ("objective", "no_objective", "milestones", "credit")

_OBJECTIVE_STAT_KEY = "aligned.junction.held"


def _default_max_junctions(env: MettaGridConfig) -> int:
    map_builder = env.game.map_builder
    if not isinstance(map_builder, MapGenConfig):
        return 1

    width = map_builder.width
    height = map_builder.height
    if width is None or height is None:
        return 1

    instance = map_builder.instance
    building_coverage = getattr(instance, "building_coverage", None)
    if not isinstance(building_coverage, (int, float)):
        return 1

    padding = 1  # UniformExtractorParams default
    spacing = padding + 1

    available_height = max(0, height - 2 * padding)
    available_width = max(0, width - 2 * padding)
    max_rows = max(0, (available_height + spacing - 1) // spacing)
    max_cols = max(0, (available_width + spacing - 1) // spacing)
    max_possible = max_rows * max_cols

    interior_width = max(0, width - 2)
    interior_height = max(0, height - 2)
    desired = int(float(building_coverage) * interior_width * interior_height)

    return min(max_possible, max(1, desired))


def _apply_milestones(rewards: AgentRewards, *, max_junctions: int) -> None:
    """Add milestone shaping rewards onto an existing baseline."""
    w_junction_aligned = 1.0
    w_scramble_act = 0.5
    w_align_act = 1.0

    rewards.collective_stats["aligned.junction"] = w_junction_aligned
    rewards.collective_stats_max["aligned.junction"] = w_junction_aligned * max_junctions

    rewards.stats["junction.scrambled_by_agent"] = w_scramble_act
    rewards.stats["junction.aligned_by_agent"] = w_align_act
    rewards.stats_max["junction.scrambled_by_agent"] = w_scramble_act * max_junctions
    rewards.stats_max["junction.aligned_by_agent"] = w_align_act * max_junctions


def _apply_credit(rewards: AgentRewards, *, max_junctions: int) -> None:
    """Add dense precursor shaping rewards onto an existing baseline."""
    w_heart = 0.05
    cap_heart = 0.5
    w_align_gear = 0.2
    cap_align_gear = 0.4
    w_scramble_gear = 0.2
    cap_scramble_gear = 0.4
    w_element_gain = 0.001
    cap_element_gain = 0.1

    rewards.stats.update(
        {
            "heart.gained": w_heart,
            "aligner.gained": w_align_gear,
            "scrambler.gained": w_scramble_gear,
            "carbon.gained": w_element_gain,
            "oxygen.gained": w_element_gain,
            "germanium.gained": w_element_gain,
            "silicon.gained": w_element_gain,
        }
    )
    rewards.stats_max.update(
        {
            "heart.gained": cap_heart,
            "aligner.gained": cap_align_gear,
            "scrambler.gained": cap_scramble_gear,
            "carbon.gained": cap_element_gain,
            "oxygen.gained": cap_element_gain,
            "germanium.gained": cap_element_gain,
            "silicon.gained": cap_element_gain,
        }
    )

    w_deposit = 0.002
    cap_deposit = 0.2
    for element in ["carbon", "oxygen", "germanium", "silicon"]:
        stat = f"collective.{element}.deposited"
        rewards.collective_stats[stat] = w_deposit
        rewards.collective_stats_max[stat] = cap_deposit


def apply_reward_variants(env: MettaGridConfig, *, variants: str | Sequence[str] | None = None) -> None:
    """Apply CogsGuard reward variants to `env`.

    Variants are stackable:
    - `objective`: no-op marker; keeps the mission's default objective reward wiring.
    - `no_objective`: disables the objective stat reward (`aligned.junction.held`).
    - `milestones`: adds shaped rewards for aligning/scrambling junctions and holding more junctions.
    - `credit`: adds additional dense shaping for precursor behaviors (resources/gear/deposits).
    """
    if not variants:
        return

    variant_names = [variants] if isinstance(variants, str) else list(variants)

    reward_variants: list[CogsGuardRewardVariant] = []
    for variant_name in variant_names:
        if variant_name not in AVAILABLE_REWARD_VARIANTS:
            available = ", ".join(AVAILABLE_REWARD_VARIANTS)
            raise ValueError(f"Unknown Cogsguard reward variant '{variant_name}'. Available: {available}")
        variant = cast(CogsGuardRewardVariant, variant_name)
        if variant in reward_variants:
            continue
        reward_variants.append(variant)

    enabled = set(reward_variants)
    if enabled <= {"objective"}:
        return

    max_junctions = _default_max_junctions(env)

    # Start from the mission's existing objective baseline to preserve its scaling.
    rewards = env.game.agent.rewards.model_copy(deep=True)
    if "no_objective" in enabled:
        rewards.collective_stats.pop(_OBJECTIVE_STAT_KEY, None)
        rewards.collective_stats_max.pop(_OBJECTIVE_STAT_KEY, None)
    if "milestones" in enabled:
        _apply_milestones(rewards, max_junctions=max_junctions)
    if "credit" in enabled:
        _apply_credit(rewards, max_junctions=max_junctions)

    env.game.agent.rewards = rewards

    # Deterministic label suffix order (exclude "objective").
    for variant in AVAILABLE_REWARD_VARIANTS:
        if variant == "objective":
            continue
        if variant in enabled:
            env.label += f".{variant}"
