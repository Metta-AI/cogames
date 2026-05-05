from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

import cogames.game as game_module
from cogames.standalone_games import STANDALONE_GAMES, GitSource, StandaloneGameInstall


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
        STANDALONE_GAMES,
        "standalone_game",
        StandaloneGameInstall(
            module_name="standalone_game.registration",
            package_name="standalone_game",
            source=GitSource(git="https://github.com/Metta-AI/standalone-game.git"),
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
        STANDALONE_GAMES,
        game_name,
        StandaloneGameInstall(
            module_name=f"{game_name}.registration",
            package_name=game_name,
            source=GitSource(git=f"https://github.com/Metta-AI/{game_name}.git"),
        ),
    )
    game_module._GAMES.pop(game_name, None)

    with pytest.raises(ValueError) as exc_info:
        game_module.get_game(game_name)

    assert f"pip install cogames[{game_name}]" in str(exc_info.value)


def test_get_game_missing_optional_dependency_has_install_hint_for_module_root_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    game_name = "standalone-game"
    module_name = "standalone_game.registration"
    monkeypatch.setitem(
        STANDALONE_GAMES,
        game_name,
        StandaloneGameInstall(
            module_name=module_name,
            package_name=game_name,
            source=GitSource(git="https://github.com/Metta-AI/standalone-game.git"),
        ),
    )
    game_module._GAMES.pop(game_name, None)
    monkeypatch.setattr(
        game_module.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(ModuleNotFoundError(name="standalone_game"))
        if name == module_name
        else None,
    )

    with pytest.raises(ValueError) as exc_info:
        game_module.get_game(game_name)

    assert f"pip install cogames[{game_name}]" in str(exc_info.value)


def test_diplomacog_is_declared_as_optional_game_module() -> None:
    standalone_game = STANDALONE_GAMES["diplomacog"]

    assert standalone_game.module_name == "diplomacog.cogame"
    assert standalone_game.package_name == "diplomacog"
    assert standalone_game.source.git == "https://github.com/Metta-AI/cogame-diplomacog.git"


def test_overcogged_is_declared_as_optional_game_module() -> None:
    standalone_game = STANDALONE_GAMES["overcogged"]

    assert standalone_game.module_name == "overcogged.game.game"
    assert standalone_game.package_name == "overcogged"
    assert standalone_game.source.git == "https://github.com/Metta-AI/cogame-overcogged.git"


def test_tribalcog_manifest_matches_package_shape() -> None:
    standalone_game = STANDALONE_GAMES["tribalcog"]

    assert standalone_game.package_name == "tribalcog"
    assert standalone_game.module_name == "tribal_village_env.recipe"
    assert standalone_game.source.git == "https://github.com/Metta-AI/cogame-tribal.git"


def test_hungercog_manifest_matches_package_shape() -> None:
    standalone_game = STANDALONE_GAMES["hungercog"]

    assert standalone_game.package_name == "hungercog"
    assert standalone_game.module_name == "hungercog.game"
    assert standalone_game.source.git == "https://github.com/Metta-AI/cogame-hungercog.git"


def test_amongcogs_manifest_matches_package_shape() -> None:
    standalone_game = STANDALONE_GAMES["amongcogs"]

    assert standalone_game.package_name == "amongcogs"
    assert standalone_game.module_name == "amongcogs.game.game"
    assert standalone_game.source.git == "https://github.com/Metta-AI/cogame-amongcogs.git"


def test_werecog_manifest_matches_package_shape() -> None:
    standalone_game = STANDALONE_GAMES["werecog"]

    assert standalone_game.package_name == "werecog"
    assert standalone_game.module_name == "werecog.cogame"
    assert standalone_game.source.git == "https://github.com/Metta-AI/cogame-werecog.git"


def test_cogsguard_manifest_matches_package_shape() -> None:
    standalone_game = STANDALONE_GAMES["cogsguard"]

    assert standalone_game.package_name == "cogsguard"
    assert standalone_game.module_name == "cogsguard.game.game"
    assert standalone_game.source.git == "https://github.com/Metta-AI/cogame-cogsguard.git"
