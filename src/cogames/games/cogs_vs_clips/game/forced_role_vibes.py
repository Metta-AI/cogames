"""ForcedRoleVibes variant: forces initial role vibes per agent."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from pydantic import Field

from cogames.core import CoGameMissionVariant
from mettagrid.config.mettagrid_config import MettaGridConfig

if TYPE_CHECKING:
    from cogames.games.cogs_vs_clips.missions.mission import CvCMission


# TODO: unchecked variant
class ForcedRoleVibesVariant(CoGameMissionVariant):
    name: str = "forced_role_vibes"
    description: str = "Force each agent's initial vibe by role using team-local agent order."

    role_order: list[str] = Field(default_factory=lambda: ["miner", "aligner", "scrambler", "scout"])
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

            role_name = self.role_order[idx % len(self.role_order)]
            agent.vibe = vibe_id_by_name[role_name]

        if self.disable_change_vibe:
            env.game.actions.change_vibe.enabled = False
