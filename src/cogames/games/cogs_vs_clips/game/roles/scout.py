"""Scout role: increased HP and energy for exploration and reconnaissance."""

from __future__ import annotations

from typing import override

from cogames.cogs_vs_clips.energy import EnergyVariant
from cogames.core import Deps
from cogames.games.cogs_vs_clips.game.damage import DamageVariant
from cogames.games.cogs_vs_clips.game.roles.role import RoleVariant
from cogames.variants import ResolvedDeps


class ScoutVariant(RoleVariant):
    """Enable the scout role: increased HP and energy."""

    name: str = "scout_role"
    description: str = "Scout role: increased HP and energy for exploration and reconnaissance."
    hp_modifier: int = 400
    energy_modifier: int = 100

    @override
    def dependencies(self) -> Deps:
        parent = super().dependencies()
        return Deps(required=parent.required, optional=[DamageVariant, EnergyVariant])

    @override
    def configure(self, deps: ResolvedDeps) -> None:
        super().configure(deps)

        d = deps.optional(DamageVariant)
        if d is not None:
            d.modifiers["scout"] = self.hp_modifier

        e = deps.optional(EnergyVariant)
        if e is not None:
            e.modifiers["scout"] = self.energy_modifier
