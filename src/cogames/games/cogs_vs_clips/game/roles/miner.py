"""Miner role: enhanced resource extraction with increased cargo capacity."""

from __future__ import annotations

from typing import override

from cogames.core import Deps
from cogames.games.cogs_vs_clips.game.cargo import CargoLimitVariant
from cogames.games.cogs_vs_clips.game.extractors import ExtractorsVariant
from cogames.games.cogs_vs_clips.game.roles.role import RoleVariant
from cogames.variants import ResolvedDeps


class MinerVariant(RoleVariant):
    """Enable the miner role: enhanced resource extraction with increased cargo capacity."""

    name: str = "miner_role"
    description: str = "Miner role: enhanced resource extraction with increased cargo capacity."

    cargo_modifier: int = 40
    extract_amount: int = 10

    @override
    def dependencies(self) -> Deps:
        parent = super().dependencies()
        return Deps(required=parent.required, optional=[CargoLimitVariant, ExtractorsVariant])

    @override
    def configure(self, deps: ResolvedDeps) -> None:
        super().configure(deps)

        cargo_limit = deps.optional(CargoLimitVariant)
        if cargo_limit is not None:
            cargo_limit.modifiers["miner"] = self.cargo_modifier

        extractors = deps.optional(ExtractorsVariant)
        if extractors is not None:
            extractors.add_extraction_handler(
                name="miner_extract",
                required_resources={"miner": 1},
                cost={},
                amount=self.extract_amount,
            )
