"""Base class for role variants."""

from __future__ import annotations

from typing import override

from cogames.core import CoGameMissionVariant, Deps
from cogames.games.cogs_vs_clips.game.gear import GearVariant
from cogames.variants import ResolvedDeps


class RoleVariant(CoGameMissionVariant):
    """Base for role variants. Registers a gear item and optionally destroys it on death."""

    @property
    def item_name(self) -> str:
        """Gear item / resource name for this role (variant name without '_role' suffix)."""
        return self.name.removesuffix("_role")

    @override
    def dependencies(self) -> Deps:
        return Deps(required=[GearVariant])

    @override
    def configure(self, deps: ResolvedDeps) -> None:
        deps.required(GearVariant).items.append(self.item_name)
