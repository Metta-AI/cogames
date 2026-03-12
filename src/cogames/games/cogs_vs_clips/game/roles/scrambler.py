"""Scrambler role: disrupts enemy junction control."""

from __future__ import annotations

from typing import override

from cogames.core import Deps
from cogames.games.cogs_vs_clips.game.damage import DamageVariant
from cogames.games.cogs_vs_clips.game.roles.role import RoleVariant
from cogames.games.cogs_vs_clips.game.teams.junction import TeamJunctionVariant
from cogames.variants import ResolvedDeps


class ScramblerVariant(RoleVariant):
    """Enable the scrambler role: disrupts enemy junction control."""

    name: str = "scrambler_role"
    description: str = "Scrambler role: disrupts enemy junction control."
    hp_modifier: int = 200

    @override
    def dependencies(self) -> Deps:
        parent = super().dependencies()
        return Deps(required=parent.required, optional=[TeamJunctionVariant, DamageVariant])

    @override
    def configure(self, deps: ResolvedDeps) -> None:
        super().configure(deps)

        tj = deps.optional(TeamJunctionVariant)
        if tj is not None:
            tj.scramble_required_resources["scrambler"] = 1

        d = deps.optional(DamageVariant)
        if d is not None:
            d.modifiers["scrambler"] = self.hp_modifier
