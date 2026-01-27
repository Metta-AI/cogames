"""Reward preset wiring for the CogsGuard (Cogs vs Clips) mission.

The mission has a single "true" objective signal, plus optional shaping variants.
These variants are mutually exclusive and are intended for different stages of learning.
"""

from __future__ import annotations

from typing import Literal, Sequence, cast

from mettagrid.config.mettagrid_config import AgentRewards, MettaGridConfig
from mettagrid.mapgen.mapgen import MapGenConfig

CogsGuardRewardVariant = Literal["credit", "milestones", "objective"]

AVAILABLE_REWARD_VARIANTS: tuple[CogsGuardRewardVariant, ...] = ("objective", "milestones", "credit")


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


def _objective_rewards(*, max_steps: int) -> AgentRewards:
    """Objective-only rewards.

    Encourages holding as many junctions as possible for as long as possible via
    the per-tick stat `aligned.junction.held` (which increments by the number of
    currently-aligned junctions each timestep).
    """
    return AgentRewards(
        collective_stats={
            "aligned.junction.held": 1.0 / max_steps,
        },
    )


def _milestone_rewards(*, max_steps: int, max_junctions: int) -> AgentRewards:
    """Milestone rewards (objective + alignment milestones).

    Adds shaping aligned with the high-level task:
    - `aligned.junction`: state-based reward for the current count of cogs-aligned junctions
      (automatic penalty if the count drops when junctions flip away).
    - `junction.aligned_by_agent` / `junction.scrambled_by_agent`: event credit for the agent
      that performed the align/scramble action.

    Caps are proportional to `max_junctions` to keep total episode returns stable as maps scale.
    """
    rewards = _objective_rewards(max_steps=max_steps)

    w_junction_aligned = 1.0
    w_scramble_act = 0.5
    w_align_act = 1.0

    rewards.collective_stats["aligned.junction"] = w_junction_aligned
    rewards.collective_stats_max["aligned.junction"] = w_junction_aligned * max_junctions

    rewards.stats = {
        "junction.scrambled_by_agent": w_scramble_act,
        "junction.aligned_by_agent": w_align_act,
    }
    rewards.stats_max = {
        "junction.scrambled_by_agent": w_scramble_act * max_junctions,
        "junction.aligned_by_agent": w_align_act * max_junctions,
    }

    return rewards


def _credit_rewards(*, max_steps: int, max_junctions: int) -> AgentRewards:
    """Credit-assignment rewards (milestones + dense precursor shaping).

    Extends `milestones` with small, capped "early learning" signals:
    - resource and heart acquisition (`*.gained`)
    - acquiring key gear like `aligner` / `scrambler`
    - depositing elements into the collective (`collective.*.deposited`)

    This is meant to help exploration and teach sub-skills that lead to junction alignment.
    """
    rewards = _milestone_rewards(max_steps=max_steps, max_junctions=max_junctions)

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

    return rewards


def apply_reward_variants(env: MettaGridConfig, *, variants: str | Sequence[str] | None = None) -> None:
    """Apply one of the CogsGuard reward presets to `env`.

    Presets:
    - `objective`: only the core objective (`aligned.junction.held`).
    - `milestones`: objective + shaped rewards for aligning/scrambling junctions and holding more junctions.
    - `credit`: milestones + additional dense shaping for precursor behaviors (resources/gear/deposits).

    Note: presets are mutually exclusive; pass exactly one.
    """
    if not variants:
        return

    variant_names = [variants] if isinstance(variants, str) else list(variants)

    reward_variants: list[CogsGuardRewardVariant] = []
    for variant_name in variant_names:
        if variant_name in reward_variants:
            continue
        if variant_name not in AVAILABLE_REWARD_VARIANTS:
            available = ", ".join(AVAILABLE_REWARD_VARIANTS)
            raise ValueError(f"Unknown Cogsguard reward variant '{variant_name}'. Available: {available}")
        reward_variants.append(cast(CogsGuardRewardVariant, variant_name))

    if len(reward_variants) > 1:
        available = ", ".join(AVAILABLE_REWARD_VARIANTS)
        raise ValueError(
            f"Cogsguard reward variants are mutually exclusive. Got {reward_variants}. Choose one of: {available}"
        )

    reward_variant = reward_variants[0]
    if reward_variant == "objective":
        return

    max_steps = env.game.max_steps
    max_junctions = _default_max_junctions(env)
    if reward_variant == "milestones":
        env.game.agent.rewards = _milestone_rewards(max_steps=max_steps, max_junctions=max_junctions)
    elif reward_variant == "credit":
        env.game.agent.rewards = _credit_rewards(max_steps=max_steps, max_junctions=max_junctions)

    env.label += f".{reward_variant}"
