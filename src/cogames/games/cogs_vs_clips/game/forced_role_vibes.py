"""ForcedRoleVibes variant: assigns per-agent roles and forces initial vibes."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from pydantic import Field

from cogames.core import CoGameMissionVariant
from mettagrid.config.game_value import inv
from mettagrid.config.mettagrid_config import MettaGridConfig

if TYPE_CHECKING:
    from cogames.games.cogs_vs_clips.missions.mission import CvCMission


# TODO: unchecked variant
class ForcedRoleVibesVariant(CoGameMissionVariant):
    name: str = "forced_role_vibes"
    description: str = "Assign per-agent roles as a global observation and force each agent's initial vibe by role."

    role_order: list[str] = Field(default_factory=lambda: ["miner", "aligner", "scrambler", "scout"])
    role_id_item: str = Field(default="role_id", description="Inventory item used for the global role_id token.")
    disable_change_vibe: bool = Field(default=True, description="Disable change_vibe so role vibes are forced.")
    per_team: bool = Field(default=True, description="Assign roles by index-within-team.")

    @override
    def modify_env(self, mission: CvCMission, env: MettaGridConfig) -> None:
        allowed_roles = {"miner", "aligner", "scrambler", "scout"}
        if not self.role_order:
            raise ValueError("role_order must be non-empty")
        unknown = [r for r in self.role_order if r not in allowed_roles]
        if unknown:
            raise ValueError(f"Unknown role(s) in role_order: {unknown}. Allowed: {sorted(allowed_roles)}")

        # Make role_id available as a resource, then add it as a per-agent global observation token.
        if self.role_id_item not in env.game.resource_names:
            env.game.resource_names = [*env.game.resource_names, self.role_id_item]

        obs_key = f"inv:own:{self.role_id_item}"
        if obs_key not in env.game.obs.global_obs.obs:
            env.game.obs.global_obs.obs[obs_key] = inv(f"agent.{self.role_id_item}")

        vibe_id_by_name = {name: idx for idx, name in enumerate(env.game.vibe_names)}
        missing_vibes = [r for r in set(self.role_order) if r not in vibe_id_by_name]
        if missing_vibes:
            raise ValueError(
                f"Missing role vibe(s) in env.game.vibe_names: {missing_vibes}. "
                "Expected role names to be present as vibe names."
            )

        # Assign roles and force initial vibe.
        counters: dict[int, int] = {}
        for agent in env.game.agents:
            if self.per_team:
                group_key: int = agent.team_id
            else:
                group_key = 0
            idx = counters.get(group_key, 0)
            counters[group_key] = idx + 1

            role_id = idx % len(self.role_order)
            role_name = self.role_order[role_id]

            agent.vibe = vibe_id_by_name[role_name]
            agent.inventory.initial = {**agent.inventory.initial, self.role_id_item: role_id}

        if self.disable_change_vibe:
            env.game.actions.change_vibe.enabled = False
