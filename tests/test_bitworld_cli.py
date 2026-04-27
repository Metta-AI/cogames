from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from cogames.cli import bitworld as bitworld_cli


def _make_bitworld_game(root: Path, name: str) -> None:
    game_dir = root / name
    game_dir.mkdir(parents=True)
    (game_dir / f"{name}.nim").write_text("", encoding="utf-8")


def test_discover_games_lists_bitworld_game_folders(tmp_path: Path) -> None:
    _make_bitworld_game(tmp_path, "fancy_cookout")
    _make_bitworld_game(tmp_path, "planet_wars")
    _make_bitworld_game(tmp_path, "global_ui")
    _make_bitworld_game(tmp_path, "overworld")
    _make_bitworld_game(tmp_path, "pufferlib")
    _make_bitworld_game(tmp_path, "tools")
    (tmp_path / "client").mkdir()
    (tmp_path / "client" / "client.nim").write_text("", encoding="utf-8")

    games = bitworld_cli.discover_games(tmp_path)

    assert [game.name for game in games] == ["fancy_cookout", "planet_wars"]
    assert games[0].title == "Fancy Cookout"
    assert games[0].entrypoint == tmp_path / "fancy_cookout" / "fancy_cookout.nim"


def test_quick_run_args_match_bitworld_launcher_flags(tmp_path: Path) -> None:
    args = bitworld_cli.quick_run_args(
        tmp_path / "tools" / "quick_run",
        bitworld_cli.QuickRunConfig(
            game="fancy_cookout",
            port=8080,
            players=4,
            address="0.0.0.0",
            save_replay=Path("run.bitreplay"),
        ),
    )

    assert args == [
        str(tmp_path / "tools" / "quick_run"),
        "fancy_cookout",
        "8080",
        "--players:4",
        "--address:0.0.0.0",
        "--save-replay:run.bitreplay",
    ]


def test_quick_run_cmd_delegates_to_installed_bitworld_launcher(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _make_bitworld_game(tmp_path, "fancy_cookout")
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "quick_run.nim").write_text("", encoding="utf-8")
    calls: list[tuple[list[str], Path]] = []

    def fake_run(args: list[str], *, cwd: Path, check: bool) -> subprocess.CompletedProcess[str]:
        assert check
        calls.append((args, cwd))
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(bitworld_cli, "bitworld_root", lambda: tmp_path)
    monkeypatch.setattr(bitworld_cli.subprocess, "run", fake_run)

    bitworld_cli.quick_run_cmd(
        "fancy_cookout",
        8080,
        players=2,
        address="127.0.0.1",
        save_replay=Path("latest.bitreplay"),
        nim="nim-test",
    )

    assert calls == [
        (["nim-test", "c", "tools/quick_run.nim"], tmp_path),
        (
            [
                str(tmp_path / "tools" / "quick_run"),
                "fancy_cookout",
                "8080",
                "--players:2",
                "--address:127.0.0.1",
                "--save-replay:latest.bitreplay",
            ],
            tmp_path,
        ),
    ]


def test_missing_bitworld_extra_has_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bitworld_cli.importlib.util, "find_spec", lambda name: None if name == "bitworld" else object())

    with pytest.raises(SystemExit) as exc_info:
        bitworld_cli.bitworld_root()

    assert "pip install cogames[bitworld]" in str(exc_info.value)
