from __future__ import annotations

from typing import TYPE_CHECKING, override

from cogames.core import CoGameMissionVariant, Deps
from cogames.games.cogs_vs_clips.game.elements import ElementsVariant
from cogames.games.cogs_vs_clips.game.extractors import ExtractorsVariant
from cogames.variants import ResolvedDeps
from mettagrid.config.filter.periodic_filter import PeriodicFilter
from mettagrid.config.game_value import GameValueRatio, QueryCountValue, val
from mettagrid.config.handler_config import Handler, query
from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.config.mutation.query_inventory_mutation import queryDelta
from mettagrid.config.tag import typeTag

if TYPE_CHECKING:
    from cogames.games.cogs_vs_clips.missions.mission import CvCMission


class EndlessVariant(CoGameMissionVariant):
    """Infinite game: no step limit, extractors persist and periodically refill."""

    name: str = "endless"
    description: str = "No max_steps, extractors never disappear, extractors periodically refill."
    refill_period: int = 1000
    refill_amount: int = 200
    refill_fraction: int = 4

    @override
    def dependencies(self) -> Deps:
        return Deps(optional=[ExtractorsVariant])

    @override
    def configure(self, deps: ResolvedDeps) -> None:
        extractors = deps.optional(ExtractorsVariant)
        if extractors:
            extractors.remove_when_empty = False

    @override
    def modify_env(self, mission: CvCMission, env: MettaGridConfig) -> None:
        env.game.max_steps = 0

        extractors = mission.optional_variant(ExtractorsVariant)
        if extractors:
            elements = mission.required_variant(ElementsVariant)
            for element in elements.elements:
                env.game.on_tick[f"{element}_extractor_refill"] = self._refill_handler(element)

    def _refill_handler(self, element: str) -> Handler:
        """Periodic handler that refills a random subset of extractors for one element."""
        key = f"{element}_extractor"
        count_gv = QueryCountValue(query=query(typeTag(key)))
        max_items_gv = GameValueRatio(count_gv, val(self.refill_fraction))
        q = query(typeTag(key))
        q.max_items = max_items_gv
        q.order_by = "random"
        return Handler(
            filters=[PeriodicFilter(period=self.refill_period)],
            mutations=[queryDelta(q, {element: self.refill_amount})],
        )
