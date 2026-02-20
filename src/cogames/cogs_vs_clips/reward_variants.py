"""Reward preset wiring for the CogsGuard (Cogs vs Clips) mission.

The mission has a single "true" objective signal, plus optional shaping variants.
Reward variants are stackable; each one adds additional shaping signals on top of the
mission's default objective rewards.
"""

from __future__ import annotations

import json
from typing import Literal, Sequence, cast

from mettagrid.config.game_value import stat
from mettagrid.config.mettagrid_config import AgentConfig, LogSumStatConfig, MettaGridConfig
from mettagrid.config.reward_config import AgentReward, reward

CogsGuardRewardVariant = Literal[
    "aligner",
    "credit",
    "milestones",
    "miner",
    "no_objective",
    "penalize_vibe_change",
    "objective",
    "role_conditional",
    "scout",
    "scrambler",
]

AVAILABLE_REWARD_VARIANTS: tuple[CogsGuardRewardVariant, ...] = (
    "objective",
    "no_objective",
    "milestones",
    "credit",
    "miner",
    "aligner",
    "scrambler",
    "scout",
    "role_conditional",
    "penalize_vibe_change",
)

_OBJECTIVE_STAT_KEY = "aligned_junction_held"
_ROLE_ORDER: tuple[str, ...] = ("miner", "aligner", "scrambler", "scout")
_ROLE_NAMES = set(_ROLE_ORDER)


def _role_name_from_vibe(env: MettaGridConfig, vibe_id: int) -> str | None:
    if vibe_id < 0 or vibe_id >= len(env.game.vibe_names):
        return None
    vibe_name = env.game.vibe_names[vibe_id]
    if vibe_name not in _ROLE_NAMES:
        return None
    return vibe_name


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


def _apply_penalize_vibe_change(rewards: dict[str, AgentReward]) -> None:
    """Add penalty for vibe changes to discourage spamming."""
    w_vibe_change = -0.01
    rewards["vibe_change_penalty"] = reward(stat("action.change_vibe.success"), weight=w_vibe_change)


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
        "aligner_lost": reward(stat("aligner.lost"), weight=-w_align_gear, max=-cap_align_gear),
        "scrambler_gained": reward(stat("scrambler.gained"), weight=w_scramble_gear, max=cap_scramble_gear),
        "scrambler_lost": reward(stat("scrambler.lost"), weight=-w_scramble_gear, max=-cap_scramble_gear),
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


def _apply_aligner(rewards: dict[str, AgentReward]) -> None:
    """Add aligner-focused shaping rewards."""
    # Aligner gear acquisition/loss (aligners are needed to align junctions)
    rewards["aligner_gained"] = reward(stat("aligner.gained"), weight=2.0)
    rewards["aligner_lost"] = reward(stat("aligner.lost"), weight=-2.0)

    # Heart acquisition/loss (hearts are consumed to align junctions)
    rewards["heart_gained"] = reward(stat("heart.gained"), weight=0.5)
    rewards["heart_lost"] = reward(stat("heart.lost"), weight=-0.5)

    # Junction alignment (the primary aligner objective)
    rewards["junction_aligned_by_agent"] = reward(stat("junction.aligned_by_agent"), weight=5.0)


_MINER_ELEMENTS = ("carbon", "oxygen", "germanium", "silicon")

_MINER_GAIN_LOG_SUM = LogSumStatConfig(
    stat_name="gain_diversity",
    stat_suffix=".gained",
    resources=list(_MINER_ELEMENTS),
)

_MINER_DEPOSIT_LOG_SUM = LogSumStatConfig(
    stat_name="deposit_diversity",
    stat_suffix=".deposited",
    resources=list(_MINER_ELEMENTS),
)


def _apply_miner(rewards: dict[str, AgentReward], agent_cfg: AgentConfig) -> None:
    """Add miner-focused shaping rewards."""
    # Gear acquisition/retention
    rewards["miner_gained"] = reward(stat("miner.gained"), weight=1.0)
    rewards["miner_lost"] = reward(stat("miner.lost"), weight=-1.0)
    rewards["heart_gained"] = reward(stat("heart.gained"), weight=-0.1)
    for other_role in ("aligner", "scout", "scrambler"):
        rewards[f"{other_role}_gained"] = reward(stat(f"{other_role}.gained"), weight=-1.0)

    # Balanced resource gain/deposit (log-product gives diminishing returns, encouraging diversity)
    agent_cfg.log_sum_stats = list(agent_cfg.log_sum_stats) + [_MINER_GAIN_LOG_SUM, _MINER_DEPOSIT_LOG_SUM]
    rewards["gain_diversity"] = reward(stat("gain_diversity"), weight=0.5)
    rewards["deposit_diversity"] = reward(stat("deposit_diversity"), weight=0.5)


def _apply_scout(rewards: dict[str, AgentReward]) -> None:
    """Add scout-focused shaping rewards."""
    # Scout gear acquisition/loss
    rewards["scout_gained"] = reward(stat("scout.gained"), weight=2.0)
    rewards["scout_lost"] = reward(stat("scout.lost"), weight=-2.0)

    rewards["cell_visited"] = reward(stat("cell.visited"), weight=0.0001)


def _apply_scrambler(rewards: dict[str, AgentReward]) -> None:
    """Add scrambler-focused shaping rewards."""
    # Scrambler gear acquisition/loss
    rewards["scrambler_gained"] = reward(stat("scrambler.gained"), weight=2.0)
    rewards["scrambler_lost"] = reward(stat("scrambler.lost"), weight=-2.0)

    # Penalize losing hearts and resources (dying loaded is costly, dying empty is cheap)
    rewards["heart_lost"] = reward(stat("heart.lost"), weight=-0.5)
    for element in ("carbon", "oxygen", "germanium", "silicon"):
        rewards[f"{element}_lost"] = reward(stat(f"{element}.lost"), weight=-0.1)

    # Junction scrambling (the primary scrambler objective)
    rewards["junction_scrambled_by_agent"] = reward(stat("junction.scrambled_by_agent"), weight=5.0)


def apply_reward_variants(env: MettaGridConfig, *, variants: str | Sequence[str] | None = None) -> None:
    """Apply CogsGuard reward variants to `env`.

    Variants are stackable:
    - `objective`: no-op marker; keeps the mission's default objective reward wiring.
    - `no_objective`: disables the objective stat reward (`junction.held`).
    - `milestones`: adds shaped rewards for aligning/scrambling junctions and holding more junctions.
    - `credit`: adds additional dense shaping for precursor behaviors (resources/gear/deposits).
    - `miner`: add miner-focused shaping rewards.
    - `aligner`: add aligner-focused shaping rewards.
    - `scrambler`: add scrambler-focused shaping rewards.
    - `scout`: add scout-focused shaping rewards.
    - `role_conditional`: apply one of the 4 role shapers per agent (Miner/Aligner/Scrambler/Scout).
    - `penalize_vibe_change`: adds a penalty for vibe changes to discourage spamming.
    """
    if not variants:
        return

    # Parse JSON-encoded list strings (e.g., '["milestones"]' from sweeps)
    if isinstance(variants, str):
        if variants.startswith("["):
            try:
                parsed = json.loads(variants)
                variant_names = list(parsed) if isinstance(parsed, list) else [variants]
            except json.JSONDecodeError:
                variant_names = [variants]
        else:
            variant_names = [variants]
    else:
        variant_names = list(variants)

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

    agent_cfgs = env.game.agents if env.game.agents else [env.game.agent]
    if "role_conditional" in enabled and not env.game.agents:
        raise ValueError("role_conditional reward variant requires env.game.agents (per-agent configs)")

    role_by_agent_idx: list[str] = []
    if "role_conditional" in enabled:
        counters: dict[str | int, int] = {}
        for agent_cfg in agent_cfgs:
            group_key: str | int = agent_cfg.collective if agent_cfg.collective is not None else agent_cfg.team_id
            idx_within_group = counters.get(group_key, 0)
            counters[group_key] = idx_within_group + 1

            role_name = _role_name_from_vibe(env, agent_cfg.vibe)
            if role_name is None:
                explicit_role_id = agent_cfg.inventory.initial.get("role_id")
                if explicit_role_id is not None:
                    role_id = int(explicit_role_id)
                else:
                    role_id = idx_within_group
                role_name = _ROLE_ORDER[role_id % len(_ROLE_ORDER)]

            role_by_agent_idx.append(role_name)

    for agent_cfg in agent_cfgs:
        rewards = dict(agent_cfg.rewards)

        if "no_objective" in enabled:
            rewards.pop(_OBJECTIVE_STAT_KEY, None)
        if "milestones" in enabled:
            _apply_milestones(rewards)
        if "credit" in enabled:
            _apply_credit(rewards)
        if "aligner" in enabled:
            _apply_aligner(rewards)
        if "miner" in enabled:
            _apply_miner(rewards, agent_cfg)
        if "scrambler" in enabled:
            _apply_scrambler(rewards)
        if "scout" in enabled:
            _apply_scout(rewards)
        if "role_conditional" in enabled:
            role = role_by_agent_idx.pop(0)
            if role == "miner":
                _apply_miner(rewards, agent_cfg)
            elif role == "aligner":
                _apply_aligner(rewards)
            elif role == "scrambler":
                _apply_scrambler(rewards)
            else:
                _apply_scout(rewards)
        if "penalize_vibe_change" in enabled:
            _apply_penalize_vibe_change(rewards)

        agent_cfg.rewards = rewards

    # Deterministic label suffix order (exclude "objective").
    for variant in AVAILABLE_REWARD_VARIANTS:
        if variant == "objective":
            continue
        if variant in enabled:
            env.label += f".{variant}"
