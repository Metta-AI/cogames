from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from cogames.cli import bitworld as bitworld_cli


def _make_bitworld_game(root: Path, name: str) -> None:
    game_dir = root / name
    game_dir.mkdir(parents=True)
    (game_dir / f"{name}.nim").write_text("", encoding="utf-8")


def _bitworld_replay_bytes(game: str = "among_them") -> bytes:
    game_bytes = game.encode("utf-8")
    return b"BITWORLD" + (2).to_bytes(2, "little") + len(game_bytes).to_bytes(2, "little") + game_bytes


def _write_quick_run(root: Path) -> None:
    tools_dir = root / "tools"
    tools_dir.mkdir()
    (tools_dir / "quick_run.nim").write_text(
        "",
        encoding="utf-8",
    )


def test_discover_games_lists_bitworld_game_folders(tmp_path: Path) -> None:
    _make_bitworld_game(tmp_path, "fancy_cookout")
    _make_bitworld_game(tmp_path, "planet_wars")
    _make_bitworld_game(tmp_path, "global_ui")
    _make_bitworld_game(tmp_path, "overworld")
    _make_bitworld_game(tmp_path, "pufferlib")
    _make_bitworld_game(tmp_path, "tools")
    (tmp_path / "clients").mkdir()
    (tmp_path / "clients" / "player_client.nim").write_text("", encoding="utf-8")

    games = bitworld_cli.discover_games(tmp_path)

    assert [game.name for game in games] == ["fancy_cookout", "planet_wars"]
    assert games[0].title == "Fancy Cookout"
    assert games[0].entrypoint == tmp_path / "fancy_cookout" / "fancy_cookout.nim"


def test_bitworld_root_uses_env_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(bitworld_cli.BITWORLD_ROOT_ENV, str(tmp_path))

    assert bitworld_cli.bitworld_root() == tmp_path


def test_bitworld_root_uses_installed_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(bitworld_cli.BITWORLD_ROOT_ENV, raising=False)
    monkeypatch.setattr(bitworld_cli.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(bitworld_cli.importlib.resources, "files", lambda name: tmp_path)

    assert bitworld_cli.bitworld_root() == tmp_path


def test_quick_run_args_match_bitworld_launcher_flags(tmp_path: Path) -> None:
    args = bitworld_cli.quick_run_args(
        tmp_path / "tools" / "quick_run",
        bitworld_cli.QuickRunConfig(
            game="fancy_cookout",
            port=8080,
            players=16,
            address="0.0.0.0",
            save_replay=Path("run.bitreplay"),
        ),
    )

    assert args == [
        str(tmp_path / "tools" / "quick_run"),
        "fancy_cookout",
        "8080",
        "--players:16",
        "--address:0.0.0.0",
        "--save-replay:run.bitreplay",
    ]


def test_bitworld_replay_game_parser_reads_game(tmp_path: Path) -> None:
    replay_path = tmp_path / "match.bitreplay"

    replay_path.write_bytes(_bitworld_replay_bytes())
    assert bitworld_cli.bitworld_replay_game_from_file(replay_path) == "among_them"


def test_global_client_url_uses_game_server_route() -> None:
    assert bitworld_cli.global_client_url("127.0.0.1", 59921) == "http://127.0.0.1:59921/client/global.html"


def test_launch_bitworld_replay_starts_server_and_global_viewer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _make_bitworld_game(tmp_path, "planet_wars")
    replay_path = tmp_path / "match.bitreplay"
    replay_path.write_bytes(_bitworld_replay_bytes("planet_wars"))
    opened_urls: list[str] = []
    popen_calls: list[tuple[list[str], Path]] = []

    class FakeProcess:
        returncode: int | None = None

        def poll(self) -> int | None:
            return self.returncode

        def wait(self) -> int:
            self.returncode = 0
            return 0

        def terminate(self) -> None:
            self.returncode = -15

        def kill(self) -> None:
            self.returncode = -9

    def fake_popen(args: list[str], *, cwd: Path) -> FakeProcess:
        popen_calls.append((args, cwd))
        return FakeProcess()

    monkeypatch.setattr(bitworld_cli, "bitworld_root", lambda: tmp_path)
    monkeypatch.setattr(bitworld_cli, "compile_game", lambda _root, game, _nim: game.entrypoint.with_suffix(""))
    monkeypatch.setattr(bitworld_cli, "_choose_port", lambda: 4567)
    monkeypatch.setattr(bitworld_cli, "_wait_for_port", lambda _host, _port, _process: None)
    monkeypatch.setattr(bitworld_cli.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(bitworld_cli.webbrowser, "open", lambda url: opened_urls.append(url))

    exit_code = bitworld_cli.launch_bitworld_replay(bitworld_cli.ReplayConfig(replay_path=replay_path))

    assert exit_code == 0
    assert popen_calls == [
        (
            [
                str(tmp_path / "planet_wars" / "planet_wars"),
                "--address:127.0.0.1",
                "--port:4567",
                f"--load-replay:{replay_path}",
            ],
            tmp_path / "planet_wars",
        )
    ]
    assert opened_urls == ["http://127.0.0.1:4567/client/global.html"]


def test_quick_run_cmd_delegates_to_installed_bitworld_launcher(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _make_bitworld_game(tmp_path, "among_them")
    _write_quick_run(tmp_path)
    calls: list[tuple[list[str], Path]] = []

    def fake_run(args: list[str], *, cwd: Path, check: bool) -> subprocess.CompletedProcess[str]:
        assert check
        calls.append((args, cwd))
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(bitworld_cli, "bitworld_root", lambda: tmp_path)
    monkeypatch.setattr(bitworld_cli.subprocess, "run", fake_run)

    bitworld_cli.quick_run_cmd(
        "among_them",
        8080,
        players=16,
        address="127.0.0.1",
        save_replay=Path("latest.bitreplay"),
        nim="nim-test",
    )

    assert calls == [
        (["nim-test", "c", "tools/quick_run.nim"], tmp_path),
        (
            [
                str(tmp_path / "tools" / "quick_run"),
                "among_them",
                "8080",
                "--players:16",
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
