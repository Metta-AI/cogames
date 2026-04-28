from __future__ import annotations

import zlib
from pathlib import Path

import pytest

from cogames import replays


def _bitworld_replay_bytes(game: str = "among_them") -> bytes:
    game_bytes = game.encode("utf-8")
    return b"BITWORLD" + (2).to_bytes(2, "little") + len(game_bytes).to_bytes(2, "little") + game_bytes


def _mettagrid_replay_bytes() -> bytes:
    return zlib.compress(b'{"game":"cogs_vs_clips"}')


def test_replay_viewer_registry_routes_by_replay_signature() -> None:
    assert replays.viewer_for_bytes(_bitworld_replay_bytes()).name == "bitworld"
    assert replays.viewer_for_bytes(_mettagrid_replay_bytes()).name == "mettascope"


@pytest.mark.parametrize("filename", ["cogs_vs_clips.json.z", "legacy.replay", "training_run.bin"])
def test_mettagrid_replay_paths_still_route_to_mettascope(tmp_path: Path, filename: str) -> None:
    replay_path = tmp_path / filename
    replay_path.write_bytes(_mettagrid_replay_bytes())

    assert replays.viewer_for_path(replay_path).name == "mettascope"


def test_launch_replay_path_uses_registered_viewer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    replay_path = tmp_path / "match.bitreplay"
    replay_path.write_bytes(_bitworld_replay_bytes())
    launched: list[object] = []

    monkeypatch.setattr(
        replays,
        "_launch_bitworld_replay",
        lambda request: launched.append(request.replay_path) or launched.append(request.duration) or 0,
    )

    exit_code = replays.launch_replay_path(replays.ReplayPathRequest(replay_path=replay_path, duration=1.5))

    assert exit_code == 0
    assert launched == [replay_path, 1.5]


def test_launch_bitworld_replay_bytes_writes_bitreplay_temp_file(monkeypatch: pytest.MonkeyPatch) -> None:
    launched: list[Path] = []
    launched_bytes: list[bytes] = []

    def fake_launch(request: replays.ReplayPathRequest) -> int:
        launched.append(request.replay_path)
        launched_bytes.append(request.replay_path.read_bytes())
        return 0

    monkeypatch.setattr(replays, "_launch_bitworld_replay", fake_launch)

    exit_code = replays.launch_replay_bytes(_bitworld_replay_bytes(), prefix="episode-test-")

    assert exit_code == 0
    assert launched
    assert launched[0].name.startswith("episode-test-")
    assert launched[0].suffix == ".bitreplay"
    assert launched_bytes == [_bitworld_replay_bytes()]
    assert not launched[0].exists()


def test_launch_mettascope_replay_bytes_writes_json_z_temp_file(monkeypatch: pytest.MonkeyPatch) -> None:
    replay_bytes = _mettagrid_replay_bytes()
    launched: list[Path] = []
    launched_bytes: list[bytes] = []

    def fake_launch(request: replays.ReplayPathRequest) -> int:
        launched.append(request.replay_path)
        launched_bytes.append(request.replay_path.read_bytes())
        return 0

    monkeypatch.setattr(replays, "_launch_mettascope_replay", fake_launch)

    exit_code = replays.launch_replay_bytes(replay_bytes, prefix="episode-test-")

    assert exit_code == 0
    assert launched
    assert launched[0].name.startswith("episode-test-")
    assert launched[0].name.endswith(".json.z")
    assert launched_bytes == [replay_bytes]
    assert not launched[0].exists()
