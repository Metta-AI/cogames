"""ForcedRoleVibes variant: forces initial role vibes per agent."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from pydantic import Field

from cogames.core import CoGameMissionVariant, Deps
from cogames.games.cogs_vs_clips.game.roles import ROLE_NAMES, assign_role_vibes, validate_role_name
from cogames.games.cogs_vs_clips.game.vibes import VibesVariant
from mettagrid.config.mettagrid_config import MettaGridConfig

if TYPE_CHECKING:
    from cogames.games.cogs_vs_clips.missions.mission import CvCMission


# TODO: unchecked variant
class ForcedRoleVibesVariant(CoGameMissionVariant):
    name: str = "forced_role_vibes"
    description: str = "Force each agent's initial vibe by role using team-local agent order."

    role_order: list[str] = Field(default_factory=lambda: list(ROLE_NAMES))
    disable_change_vibe: bool = Field(default=True, description="Disable change_vibe so role vibes are forced.")
    per_team: bool = Field(default=True, description="Assign roles by index-within-team.")

    @override
    def dependencies(self) -> Deps:
        return Deps(optional=[VibesVariant])

    @override
    def modify_env(self, mission: CvCMission, env: MettaGridConfig) -> None:
        if not self.role_order:
            raise ValueError("role_order must be non-empty")
        validated_role_order = tuple(validate_role_name(role_name) for role_name in self.role_order)
        counters: dict[int, int] = {}
        role_names_by_agent: list[str] = []
        for agent in env.game.agents:
            group_key = agent.team_id if self.per_team else 0
            idx = counters.get(group_key, 0)
            counters[group_key] = idx + 1
            role_names_by_agent.append(validated_role_order[idx % len(validated_role_order)])
        assign_role_vibes(env, role_names_by_agent)

        if self.disable_change_vibe:
            env.game.actions.change_vibe.enabled = False
            env.game.actions.change_vibe.vibes = []
