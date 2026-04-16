from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

import cogames.game as game_module


def test_get_game_imports_optional_standalone_game(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package_dir = tmp_path / "standalone_game"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("")
    (package_dir / "registration.py").write_text(
        textwrap.dedent(
            """
            from cogames.game import CoGame, register_game

            register_game(CoGame(name="standalone_game", missions=[], variants=[]))
            """
        )
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setitem(
        game_module._OPTIONAL_GAME_MODULES,
        "standalone_game",
        game_module.OptionalGameModule(
            module_name="standalone_game.registration",
            package_name="standalone_game",
            extra_name="standalone_game",
        ),
    )
    game_module._GAMES.pop("standalone_game", None)
    sys.modules.pop("standalone_game.registration", None)
    sys.modules.pop("standalone_game", None)

    try:
        game = game_module.get_game("standalone_game")
    finally:
        game_module._GAMES.pop("standalone_game", None)
        sys.modules.pop("standalone_game.registration", None)
        sys.modules.pop("standalone_game", None)

    assert game.name == "standalone_game"


def test_get_game_missing_optional_dependency_has_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    game_name = "missing_standalone_game"
    monkeypatch.setitem(
        game_module._OPTIONAL_GAME_MODULES,
        game_name,
        game_module.OptionalGameModule(
            module_name=f"{game_name}.registration",
            package_name=game_name,
            extra_name=game_name,
        ),
    )
    game_module._GAMES.pop(game_name, None)

    with pytest.raises(ValueError) as exc_info:
        game_module.get_game(game_name)

    assert f"pip install cogames[{game_name}]" in str(exc_info.value)


def test_overcogged_is_declared_as_optional_game_module() -> None:
    optional_game = game_module._OPTIONAL_GAME_MODULES["overcogged"]

    assert optional_game.module_name == "overcogged.game.game"
    assert optional_game.package_name == "overcogged"
    assert optional_game.extra_name == "overcogged"
