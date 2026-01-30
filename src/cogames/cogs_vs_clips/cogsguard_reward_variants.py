"""Reward preset wiring for the CogsGuard (Cogs vs Clips) mission.

The mission has a single "true" objective signal, plus optional shaping variants.
Reward variants are stackable; each one adds additional shaping signals on top of the
mission's default objective rewards.
"""

from __future__ import annotations

from typing import Literal, Sequence, cast

from mettagrid.config.game_value import stat
from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.config.reward_config import AgentReward, reward

CogsGuardRewardVariant = Literal["credit", "milestones", "no_objective", "objective"]

AVAILABLE_REWARD_VARIANTS: tuple[CogsGuardRewardVariant, ...] = ("objective", "no_objective", "milestones", "credit")

_OBJECTIVE_STAT_KEY = "aligned_junction_held"


def _apply_milestones(rewards: dict[str, AgentReward], *, max_junctions: int = 100) -> None:
    """Add milestone shaping rewards onto an existing baseline.

    Args:
        rewards: Rewards dict to modify in-place.
        max_junctions: Maximum expected number of junctions for capping rewards.
            Defaults to 100 as a reasonable upper bound for most maps.
    """
    w_junction_aligned = 1.0
    w_scramble_act = 0.5
    w_align_act = 1.0

    # Max caps based on expected junction counts
    max_junction_aligned = w_junction_aligned * max_junctions
    max_scramble = w_scramble_act * max_junctions
    max_align = w_align_act * max_junctions

    rewards["aligned_junctions"] = reward(
        stat("collective.junction"),
        weight=w_junction_aligned,
        max=max_junction_aligned,
    )

    rewards["junction_scrambled_by_agent"] = reward(
        stat("junction.scrambled_by_agent"),
        weight=w_scramble_act,
        max=max_scramble,
    )
    rewards["junction_aligned_by_agent"] = reward(
        stat("junction.aligned_by_agent"),
        weight=w_align_act,
        max=max_align,
    )


def _apply_credit(rewards: dict[str, AgentReward]) -> None:
    """Add dense precursor shaping rewards onto an existing baseline."""
    w_heart = 0.05
    cap_heart = 0.5
    w_align_gear = 0.2
    cap_align_gear = 0.4
    w_scramble_gear = 0.2
    cap_scramble_gear = 0.4
    w_element_gain = 0.001
    cap_element_gain = 0.1

    # Stats rewards for gains as a single map
    gain_rewards: dict[str, AgentReward] = {
        "heart_gained": reward(stat("heart.gained"), weight=w_heart, max=cap_heart),
        "aligner_gained": reward(stat("aligner.gained"), weight=w_align_gear, max=cap_align_gear),
        "scrambler_gained": reward(stat("scrambler.gained"), weight=w_scramble_gear, max=cap_scramble_gear),
        "carbon_gained": reward(stat("carbon.gained"), weight=w_element_gain, max=cap_element_gain),
        "oxygen_gained": reward(stat("oxygen.gained"), weight=w_element_gain, max=cap_element_gain),
        "germanium_gained": reward(stat("germanium.gained"), weight=w_element_gain, max=cap_element_gain),
        "silicon_gained": reward(stat("silicon.gained"), weight=w_element_gain, max=cap_element_gain),
    }
    rewards.update(gain_rewards)

    # Collective deposit rewards
    w_deposit = 0.002
    cap_deposit = 0.2
    deposit_rewards: dict[str, AgentReward] = {
        f"collective_{element}_deposited": reward(
            stat(f"collective.{element}.deposited"), weight=w_deposit, max=cap_deposit
        )
        for element in ["carbon", "oxygen", "germanium", "silicon"]
    }
    rewards.update(deposit_rewards)


def apply_reward_variants(env: MettaGridConfig, *, variants: str | Sequence[str] | None = None) -> None:
    """Apply CogsGuard reward variants to `env`.

    Variants are stackable:
    - `objective`: no-op marker; keeps the mission's default objective reward wiring.
    - `no_objective`: disables the objective stat reward (`junction.held`).
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

    # Start from the mission's existing objective baseline to preserve its scaling.
    rewards = dict(env.game.agent.rewards)

    if "no_objective" in enabled:
        rewards.pop(_OBJECTIVE_STAT_KEY, None)
    if "milestones" in enabled:
        _apply_milestones(rewards)
    if "credit" in enabled:
        _apply_credit(rewards)

    env.game.agent.rewards = rewards

    # Deterministic label suffix order (exclude "objective").
    for variant in AVAILABLE_REWARD_VARIANTS:
        if variant == "objective":
            continue
        if variant in enabled:
            env.label += f".{variant}"
