"""Team junction variant: makes junctions alignable by teams."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from pydantic import Field

from cogames.core import CoGameMissionVariant, Deps
from cogames.games.cogs_vs_clips.game.junction import JunctionVariant
from cogames.games.cogs_vs_clips.game.teams.hub import TeamHubVariant
from cogames.games.cogs_vs_clips.game.teams.team import TeamVariant
from mettagrid.config.filter import (
    actorHasTag,
    anyOf,
    hasTag,
    isNear,
    isNot,
)
from mettagrid.config.handler_config import Handler, actorHas, updateActor
from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.config.mutation import addTag, recomputeMaterializedQuery, removeTag
from mettagrid.config.mutation.stats_mutation import logActorAgentStat, logStatToGame
from mettagrid.config.query import query
from mettagrid.config.render_config import RenderAsset
from mettagrid.config.tag import typeTag

if TYPE_CHECKING:
    from cogames.games.cogs_vs_clips.missions.mission import CvCMission


def _neg(d: dict[str, int]) -> dict[str, int]:
    return {k: -v for k, v in d.items()}


class TeamJunctionVariant(CoGameMissionVariant):
    """Makes junctions alignable by teams: adds align/scramble handlers and tag-remove hooks."""

    name: str = "team_junction"
    description: str = "Junctions can be aligned to a team."

    align_required_resources: dict[str, int] = Field(
        default_factory=dict, description="Required items to align, set by role variants."
    )
    align_cost: dict[str, int] = Field(default_factory=dict, description="Consumed items on align, set by mission.")

    junction_align_radius: int = Field(default=15, description="Radius for junction align handlers.")
    hub_align_radius: int = Field(default=25, description="Radius for hub align handlers.")

    scramble_required_resources: dict[str, int] = Field(
        default_factory=dict, description="Required items to scramble, set by role variants."
    )
    scramble_cost: dict[str, int] = Field(
        default_factory=dict, description="Consumed items on scramble, set by mission."
    )

    @override
    def dependencies(self) -> Deps:
        return Deps(required=[JunctionVariant, TeamHubVariant])

    @override
    def modify_env(self, mission: CvCMission, env: MettaGridConfig) -> None:
        junction = env.game.objects["junction"]

        team_v = mission.required_variant(TeamVariant)
        all_teams = list(team_v.teams.values())

        junction.on_tag_remove = {
            t.net_tag(): Handler(filters=[], mutations=[removeTag(t.team_tag())]) for t in all_teams
        }

        for t in all_teams:
            scramble_filters: list = [
                hasTag(t.team_tag()),
                isNot(actorHasTag(t.team_tag())),
                actorHas({**self.scramble_required_resources, **self.scramble_cost}),
            ]
            scramble_mutations: list = [
                updateActor(_neg(self.scramble_cost)),
                removeTag(t.net_tag()),
                logActorAgentStat("junction.scrambled_by_agent"),
                logStatToGame(f"{t.name}/aligned.junction.lost"),
                recomputeMaterializedQuery(t.net_tag()),
            ]
            junction.on_use_handlers[f"scramble_{t.name}"] = Handler(
                filters=scramble_filters,
                mutations=scramble_mutations,
            )

            # Only create align handlers for teams with agents.
            if t.num_agents == 0:
                continue

            align_filters: list = [
                actorHasTag(t.team_tag()),
                isNot(hasTag(t.team_tag())),
                actorHas({**self.align_required_resources, **self.align_cost}),
                anyOf(
                    [
                        isNear(query(t.net_tag()), radius=self.junction_align_radius),
                        isNear(query(typeTag("hub"), hasTag(t.team_tag())), radius=self.hub_align_radius),
                    ]
                ),
            ]
            align_mutations: list = [
                updateActor(_neg(self.align_cost)),
                logActorAgentStat("junction.aligned_by_agent"),
                logStatToGame(f"{t.name}/aligned.junction.gained"),
                addTag(t.team_tag()),
                addTag(t.net_tag()),
                recomputeMaterializedQuery(t.net_tag()),
            ]

            junction.on_use_handlers[f"align_{t.name}"] = Handler(
                filters=align_filters,
                mutations=align_mutations,
            )

        env.game.render.assets["junction"] = [
            RenderAsset(asset="junction.working", tags=[t.team_tag()]) for t in all_teams
        ] + [RenderAsset(asset="junction")]
