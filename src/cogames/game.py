"""Game management and discovery for CoGames."""

from __future__ import annotations

import importlib
from typing import Sequence

from cogames.core import CoGameMission, CoGameMissionVariant
from cogames.standalone_games import STANDALONE_GAMES
from cogames.variants import VariantRegistry


class CoGame:
    """Base class for CoGames. Holds missions and a variant registry."""

    name: str
    missions: list[CoGameMission]
    variant_registry: VariantRegistry
    eval_missions: list[CoGameMission]

    def __init__(
        self,
        name: str,
        missions: Sequence[CoGameMission],
        variants: Sequence[CoGameMissionVariant],
        eval_missions: Sequence[CoGameMission] | None = None,
    ) -> None:
        self.name = name
        self.missions = list(missions)
        self.variant_registry = VariantRegistry(list(variants))
        self.eval_missions = list(eval_missions) if eval_missions else []


_GAMES: dict[str, "CoGame"] = {}
_GAME_MODULES: dict[str, str] = {
    "cogs_vs_clips": "cogames.games.cogs_vs_clips.game.game",
}


def _import_standalone_game(name: str) -> bool:
    if name not in STANDALONE_GAMES:
        return False
    standalone_game = STANDALONE_GAMES[name]

    try:
        importlib.import_module(standalone_game.module_name)
    except ModuleNotFoundError as exc:
        if exc.name == standalone_game.package_name:
            raise ValueError(
                f"Game '{name}' is not installed. Install it with:\n  pip install cogames[{name}]"
            ) from exc
        raise

    return True


def _ensure_game_loaded(name: str) -> None:
    if name in _GAMES:
        return
    if name in _GAME_MODULES:
        importlib.import_module(_GAME_MODULES[name])
        return
    _import_standalone_game(name)


def get_game(name: str) -> "CoGame":
    """Get a registered game by name."""
    _ensure_game_loaded(name)
    if name not in _GAMES:
        available = sorted({*_GAME_MODULES, *STANDALONE_GAMES, *_GAMES})
        raise ValueError(f"Unknown game '{name}'. Available: {', '.join(available)}")
    return _GAMES[name]


def register_game(game: "CoGame") -> None:
    """Register a game for CLI resolution."""
    _GAMES[game.name] = game
