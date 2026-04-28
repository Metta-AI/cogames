"""Replay viewer dispatch for CoGames replay files.

MettaGrid cogs/clips replays run in MettaScope. BitWorld replays run in the
BitWorld global client.
"""

from __future__ import annotations

import importlib.util
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from cogames.cli.base import console
from cogames.cli.bitworld import BITREPLAY_MAGIC, ReplayConfig, launch_bitworld_replay

REPLAY_SIGNATURE_BYTES = 64


class ReplayPathRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    replay_path: Path
    duration: float | None = Field(default=None, ge=0)


class ReplayViewer(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    name: str = Field(min_length=1)
    temp_suffix: str = Field(min_length=1)
    matches_bytes: Callable[[bytes], bool]
    launch_path: Callable[[ReplayPathRequest], int]


def _resolve_mettascope_script() -> Path:
    spec = importlib.util.find_spec("mettagrid")
    if spec is None or spec.origin is None:
        raise FileNotFoundError("mettagrid package is not available; cannot locate MettaScope.")

    package_dir = Path(spec.origin).resolve().parent
    search_roots = (package_dir, *package_dir.parents)

    for root in search_roots:
        candidate = root / "nim" / "mettascope" / "src" / "mettascope.nim"
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"MettaScope sources not found relative to installed mettagrid package (searched from {package_dir})."
    )


def _launch_mettascope_replay(request: ReplayPathRequest) -> int:
    mettascope_path = _resolve_mettascope_script()
    console.print(f"[cyan]Launching MettaScope to replay: {request.replay_path}[/cyan]")
    subprocess.run(["nim", "r", str(mettascope_path), f"--replay:{request.replay_path}"], check=True)
    return 0


def _launch_bitworld_replay(request: ReplayPathRequest) -> int:
    return launch_bitworld_replay(ReplayConfig(replay_path=request.replay_path, duration=request.duration))


def replay_viewers() -> tuple[ReplayViewer, ...]:
    return (
        ReplayViewer(
            name="bitworld",
            temp_suffix=".bitreplay",
            matches_bytes=lambda replay_bytes: replay_bytes.startswith(BITREPLAY_MAGIC),
            launch_path=_launch_bitworld_replay,
        ),
        ReplayViewer(
            name="mettascope",
            temp_suffix=".json.z",
            matches_bytes=lambda _replay_bytes: True,
            launch_path=_launch_mettascope_replay,
        ),
    )


def viewer_for_bytes(replay_bytes: bytes) -> ReplayViewer:
    for viewer in replay_viewers():
        if viewer.matches_bytes(replay_bytes):
            return viewer
    raise RuntimeError("No replay viewer registered")


def viewer_for_path(replay_path: Path) -> ReplayViewer:
    with replay_path.open("rb") as replay_file:
        return viewer_for_bytes(replay_file.read(REPLAY_SIGNATURE_BYTES))


def launch_replay_path(request: ReplayPathRequest) -> int:
    return viewer_for_path(request.replay_path).launch_path(request)


def launch_replay_bytes(replay_bytes: bytes, *, prefix: str) -> int:
    viewer = viewer_for_bytes(replay_bytes)
    with tempfile.NamedTemporaryFile(suffix=viewer.temp_suffix, delete=False, prefix=prefix) as replay_file:
        replay_file.write(replay_bytes)
        replay_path = Path(replay_file.name)
    try:
        return viewer.launch_path(ReplayPathRequest(replay_path=replay_path))
    finally:
        replay_path.unlink(missing_ok=True)
