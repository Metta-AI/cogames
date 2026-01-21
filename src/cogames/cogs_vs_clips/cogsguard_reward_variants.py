from __future__ import annotations

from typing import Literal, Sequence, cast

from mettagrid.config.mettagrid_config import AgentRewards, MettaGridConfig

CogsGuardRewardVariant = Literal["credit", "milestones", "objective"]

AVAILABLE_REWARD_VARIANTS: tuple[CogsGuardRewardVariant, ...] = ("objective", "milestones", "credit")


def _objective_rewards(*, max_steps: int) -> AgentRewards:
    return AgentRewards(
        collective_stats={
            "aligned.junction.held": 1.0 / max_steps,
        },
    )


def _milestone_rewards(*, max_steps: int) -> AgentRewards:
    rewards = _objective_rewards(max_steps=max_steps)

    w_heart = 0.02
    cap_heart = 0.2
    w_align_gear = 0.05
    cap_align_gear = 0.1
    w_scramble_gear = 0.05
    cap_scramble_gear = 0.1

    rewards.stats = {
        "heart.gained": w_heart,
        "aligner.gained": w_align_gear,
        "scrambler.gained": w_scramble_gear,
    }
    rewards.stats_max = {
        "heart.gained": cap_heart,
        "aligner.gained": cap_align_gear,
        "scrambler.gained": cap_scramble_gear,
    }

    w_deposit = 0.0005
    cap_deposit = 0.02
    for element in ["carbon", "oxygen", "germanium", "silicon"]:
        stat = f"collective.{element}.deposited"
        rewards.collective_stats[stat] = w_deposit
        rewards.collective_stats_max[stat] = cap_deposit

    return rewards


def _credit_rewards(*, max_steps: int) -> AgentRewards:
    rewards = _milestone_rewards(max_steps=max_steps)

    w_scramble_act = 0.2
    cap_scramble_act = 0.2
    w_align_act = 0.2
    cap_align_act = 0.2

    rewards.stats.update(
        {
            "junction.scrambled_by_agent": w_scramble_act,
            "junction.aligned_by_agent": w_align_act,
        }
    )
    rewards.stats_max.update(
        {
            "junction.scrambled_by_agent": cap_scramble_act,
            "junction.aligned_by_agent": cap_align_act,
        }
    )

    return rewards


def apply_reward_variants(env: MettaGridConfig, *, variants: str | Sequence[str] | None = None) -> None:
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
    if reward_variant == "milestones":
        env.game.agent.rewards = _milestone_rewards(max_steps=max_steps)
    elif reward_variant == "credit":
        env.game.agent.rewards = _credit_rewards(max_steps=max_steps)

    env.label += f".{reward_variant}"
