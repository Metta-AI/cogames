"""Aligner role: converts neutral junctions to team-owned."""

from __future__ import annotations

from typing import override

from cogames.core import Deps
from cogames.games.cogs_vs_clips.game.roles.role import RoleVariant
from cogames.games.cogs_vs_clips.game.teams.junction import TeamJunctionVariant
from cogames.variants import ResolvedDeps


class AlignerVariant(RoleVariant):
    """Enable the aligner role: converts neutral junctions to team-owned."""

    name: str = "aligner_role"
    description: str = "Aligner role: converts neutral junctions to team-owned."

    @override
    def dependencies(self) -> Deps:
        parent = super().dependencies()
        return Deps(required=parent.required, optional=[TeamJunctionVariant])

    @override
    def configure(self, deps: ResolvedDeps) -> None:
        super().configure(deps)

        tj = deps.optional(TeamJunctionVariant)
        if tj is not None:
            tj.align_required_resources["aligner"] = 1
