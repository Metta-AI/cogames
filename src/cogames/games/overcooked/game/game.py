"""CoGame registration for Overcooked in the future packaged-games layout."""

from __future__ import annotations

from cogames.core import CoGameMission
from cogames.game import CoGame, register_game
from cogames.games.overcooked.game import load_variants
from cogames.games.overcooked.missions import make_basic_mission
from cogames.variants import VariantRegistry


class OvercookedCoGame(CoGame):
    def __init__(self) -> None:
        self.name = "overcooked"
        self._missions: list[CoGameMission] | None = None
        self._variant_registry: VariantRegistry | None = None
        self._eval_missions: list[CoGameMission] = []

    def _ensure_loaded(self) -> None:
        if self._variant_registry is not None:
            return

        _, _, variants = load_variants()
        self._missions = [make_basic_mission()]
        self._variant_registry = VariantRegistry(list(variants))

    @property
    def missions(self) -> list[CoGameMission]:
        self._ensure_loaded()
        missions = self._missions
        assert missions is not None
        return missions

    @property
    def variant_registry(self) -> VariantRegistry:
        self._ensure_loaded()
        variant_registry = self._variant_registry
        assert variant_registry is not None
        return variant_registry

    @property
    def eval_missions(self) -> list[CoGameMission]:
        return self._eval_missions


register_game(OvercookedCoGame())
