"""Re-export shim for game registration with cogames-CLI lazy install.

`CoGame` and `register_game` are re-exported from `mettagrid.cogame.game`,
so the in-process game registry (`_GAMES`) is shared with anything that
imports `mettagrid.cogame`. `get_game` here adds the cogames-CLI
lazy-install behavior on top of the shared registry; this lazy-install
goes away when the cogames package is deleted in PR 6.
"""

from __future__ import annotations

import importlib

from cogames.standalone_games import STANDALONE_GAMES
from mettagrid.cogame.game import _GAMES as _GAMES  # share the registry
from mettagrid.cogame.game import CoGame, register_game

__all__ = ["CoGame", "get_game", "register_game"]


def _import_standalone_game(name: str) -> bool:
    if name not in STANDALONE_GAMES:
        return False
    standalone_game = STANDALONE_GAMES[name]
    import_root = standalone_game.module_name.split(".", 1)[0]

    try:
        importlib.import_module(standalone_game.module_name)
    except ModuleNotFoundError as exc:
        if exc.name in {standalone_game.package_name, import_root}:
            raise ValueError(
                f"Game '{name}' is not installed. Install it with:\n  pip install cogames[{name}]"
            ) from exc
        raise

    return True


def _ensure_game_loaded(name: str) -> None:
    if name in _GAMES:
        return
    _import_standalone_game(name)


def get_game(name: str) -> "CoGame":
    """Get a registered game by name; lazy-imports standalone games."""
    _ensure_game_loaded(name)
    if name not in _GAMES:
        available = sorted({*STANDALONE_GAMES, *_GAMES})
        raise ValueError(f"Unknown game '{name}'. Available: {', '.join(available)}")
    return _GAMES[name]
