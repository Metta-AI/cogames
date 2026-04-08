from typing import Sequence

from cogames.games.cogs_vs_clips.game.roles.aligner import AlignerVariant
from cogames.games.cogs_vs_clips.game.roles.miner import MinerVariant
from cogames.games.cogs_vs_clips.game.roles.role import RoleVariant
from cogames.games.cogs_vs_clips.game.roles.scout import ScoutVariant
from cogames.games.cogs_vs_clips.game.roles.scrambler import ScramblerVariant
from mettagrid.config.mettagrid_config import AgentConfig, MettaGridConfig

ROLE_NAMES: tuple[str, ...] = ("miner", "aligner", "scrambler", "scout")
ROLE_NAME_SET = frozenset(ROLE_NAMES)


def validate_role_name(role_name: str) -> str:
    if role_name not in ROLE_NAME_SET:
        raise ValueError(f"Unknown role {role_name!r}. Expected one of {list(ROLE_NAMES)}")
    return role_name


def assign_role_vibes(
    env: MettaGridConfig,
    role_names_by_agent: Sequence[str],
    *,
    agent_cfgs: Sequence[AgentConfig] | None = None,
) -> None:
    resolved_agent_cfgs = env.game.agents if agent_cfgs is None else list(agent_cfgs)
    if len(resolved_agent_cfgs) != len(role_names_by_agent):
        raise ValueError(
            f"Expected one role per agent, got {len(role_names_by_agent)} roles for {len(resolved_agent_cfgs)} agents"
        )
    validated_role_names = tuple(validate_role_name(role_name) for role_name in role_names_by_agent)
    vibe_id_by_name = {name: idx for idx, name in enumerate(env.game.vibe_names)}
    missing_vibes = [role_name for role_name in set(validated_role_names) if role_name not in vibe_id_by_name]
    if missing_vibes:
        raise ValueError(
            f"Missing role vibe(s) in env.game.vibe_names: {missing_vibes}. "
            "Expected role names to be present as vibe names."
        )
    for agent_cfg, role_name in zip(resolved_agent_cfgs, validated_role_names, strict=True):
        agent_cfg.vibe = vibe_id_by_name[role_name]


def require_role_names_from_vibes(env: MettaGridConfig, agent_cfgs: Sequence[AgentConfig]) -> tuple[str, ...]:
    role_names_by_agent: list[str] = []
    missing_agent_indices: list[int] = []
    for agent_idx, agent_cfg in enumerate(agent_cfgs):
        if agent_cfg.vibe < 0 or agent_cfg.vibe >= len(env.game.vibe_names):
            missing_agent_indices.append(agent_idx)
            continue
        vibe_name = env.game.vibe_names[agent_cfg.vibe]
        if vibe_name not in ROLE_NAME_SET:
            missing_agent_indices.append(agent_idx)
            continue
        role_names_by_agent.append(vibe_name)
    if missing_agent_indices:
        raise ValueError(
            "role_conditional reward variant requires explicit role vibes on every agent; "
            f"missing role vibes for agents {missing_agent_indices}"
        )
    return tuple(role_names_by_agent)


__all__ = [
    "AlignerVariant",
    "MinerVariant",
    "ROLE_NAME_SET",
    "ROLE_NAMES",
    "assign_role_vibes",
    "RoleVariant",
    "ScoutVariant",
    "ScramblerVariant",
    "require_role_names_from_vibes",
    "validate_role_name",
]
