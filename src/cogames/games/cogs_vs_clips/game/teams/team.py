"""Team configuration and variant for CvC missions."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from pydantic import Field

from cogames.core import CoGameMissionVariant
from cogames.variants import ResolvedDeps
from mettagrid.base_config import Config
from mettagrid.config.filter import hasTag
from mettagrid.config.mettagrid_config import AgentConfig, MettaGridConfig
from mettagrid.config.query import Query, query

if TYPE_CHECKING:
    from cogames.games.cogs_vs_clips.missions.mission import CvCMission


class TeamConfig(Config):
    """Configuration for a team (cogs or clips)."""

    name: str = Field(default="cogs", description="Team name used for tags and team identity")
    short_name: str = Field(default="c", description="Short prefix used for map object names")
    team_id: int = Field(default=0, description="Numeric id for this team (set when building game config)")
    num_agents: int = Field(default=8, ge=0, description="Number of agents in the team")

    def team_tag(self) -> str:
        return f"team:{self.name}"

    def net_tag(self) -> str:
        return f"net:{self.name}"

    def network_seed_query(self) -> Query:
        return query("type:hub", hasTag(self.team_tag()))


class TeamVariant(CoGameMissionVariant):
    """Set up teams and assign agents.

    When used as a dependency (default config), initializes the default cogs team.
    When used with explicit team sizes, adjusts existing team agent counts.
    """

    name: str = "team"
    description: str = "Configure teams and agent assignments."
    default_teams: dict[str, TeamConfig] = Field(default_factory=lambda: {"cogs": TeamConfig(name="cogs")})
    team_sizes: dict[str, int] = Field(default_factory=dict, description="Team name → agent count overrides.")
    teams: dict[str, TeamConfig] = Field(default_factory=dict, exclude=True)

    _team_by_id: dict[int, TeamConfig] = {}

    @override
    def configure(self, deps: ResolvedDeps) -> None:
        if not self.teams:
            self.teams = {name: t.model_copy() for name, t in self.default_teams.items()}

    def team_name(self, team_id: int) -> str | None:
        t = self._team_by_id.get(team_id)
        return t.name if t else None

    @override
    def modify_env(self, mission: CvCMission, env: MettaGridConfig) -> None:
        if self.team_sizes:
            for team_name, num_cogs in self.team_sizes.items():
                team = self.teams.get(team_name)
                if team is None:
                    raise ValueError(f"Unknown team '{team_name}'. Available: {list(self.teams.keys())}")
                team.num_agents = num_cogs
        else:
            player_teams = [t for t in self.teams.values() if t.num_agents > 0]
            if len(player_teams) == 1:
                player_teams[0].num_agents = mission.num_agents

        for i, t in enumerate(self.teams.values()):
            t.team_id = i
            self._team_by_id[i] = t

        # Assign teams to existing agents (preserving prior modifications like
        # inventory limits), adding or trimming agents only if the count changed.
        total_needed = sum(t.num_agents for t in self.teams.values())
        while len(env.game.agents) < total_needed:
            env.game.agents.append(AgentConfig())
        env.game.agents = env.game.agents[:total_needed]

        idx = 0
        for t in self.teams.values():
            tag = t.team_tag()
            for _ in range(t.num_agents):
                agent = env.game.agents[idx]
                agent.team_id = t.team_id
                if tag not in agent.tags:
                    agent.tags.append(tag)
                idx += 1

        env.game.num_agents = len(env.game.agents)
        env.game.tags.extend([t.team_tag() for t in self.teams.values()])
        env.game.tags.extend([t.net_tag() for t in self.teams.values()])
