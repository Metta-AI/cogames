"""BitWorld CLI bridge for installed websocket games."""

from __future__ import annotations

import importlib.resources
import importlib.util
import os
import subprocess
from pathlib import Path
from typing import Annotated

import typer
from pydantic import BaseModel, ConfigDict, Field
from rich.table import Table

from cogames.cli.base import console

BITWORLD_EXTRA = "bitworld"
DEFAULT_ADDRESS = "127.0.0.1"
SKIP_FOLDERS = frozenset(
    {
        "__pycache__",
        "client",
        "common",
        "dist",
        "docs",
        "global_client",
        "global_ui",
        "overworld",
        "player_client",
        "pufferlib",
        "reward_client",
        "tools",
    }
)


class BitWorldGame(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    title: str = Field(min_length=1)
    entrypoint: Path


class QuickRunConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    game: str = Field(min_length=1)
    port: int | None = Field(default=None, ge=1, le=65535)
    players: int = Field(default=1, ge=1)
    address: str = Field(default=DEFAULT_ADDRESS, min_length=1)
    save_replay: Path | None = None


bitworld_app = typer.Typer(
    help="Run optional BitWorld websocket games.",
    context_settings={"help_option_names": ["-h", "--help"]},
    no_args_is_help=True,
    rich_markup_mode="rich",
)


def bitworld_root() -> Path:
    if importlib.util.find_spec("bitworld") is None:
        raise SystemExit(
            "'cogames bitworld' requires the optional BitWorld package.\n"
            "\n"
            "Install it with:\n"
            f"  pip install cogames[{BITWORLD_EXTRA}]\n"
        )
    root = importlib.resources.files("bitworld")
    if not isinstance(root, Path):
        raise SystemExit("'cogames bitworld' requires BitWorld to be installed as a filesystem package.")
    return root


def _title_for(name: str) -> str:
    return " ".join(part.capitalize() for part in name.replace("-", "_").split("_") if part)


def discover_games(root: Path | None = None) -> list[BitWorldGame]:
    package_root = root if root is not None else bitworld_root()
    games: list[BitWorldGame] = []
    for child in sorted(package_root.iterdir(), key=lambda path: path.name):
        if not child.is_dir() or child.name.startswith(".") or child.name in SKIP_FOLDERS:
            continue
        entrypoint = child / f"{child.name}.nim"
        if entrypoint.exists():
            games.append(BitWorldGame(name=child.name, title=_title_for(child.name), entrypoint=entrypoint))
    return games


def _require_game(root: Path, game: str) -> BitWorldGame:
    games_by_name = {candidate.name: candidate for candidate in discover_games(root)}
    if game not in games_by_name:
        available = ", ".join(sorted(games_by_name))
        raise SystemExit(f"Unknown BitWorld game '{game}'. Available: {available}")
    return games_by_name[game]


def _quick_run_binary(root: Path) -> Path:
    suffix = ".exe" if os.name == "nt" else ""
    return root / "tools" / f"quick_run{suffix}"


def compile_quick_run(root: Path, nim: str) -> Path:
    subprocess.run([nim, "c", "tools/quick_run.nim"], cwd=root, check=True)
    return _quick_run_binary(root)


def quick_run_args(executable: Path, config: QuickRunConfig) -> list[str]:
    args = [str(executable), config.game]
    if config.port is not None:
        args.append(str(config.port))
    args.append(f"--players:{config.players}")
    args.append(f"--address:{config.address}")
    if config.save_replay is not None:
        args.append(f"--save-replay:{config.save_replay}")
    return args


@bitworld_app.command("games", help="List installed BitWorld games.")
def games_cmd() -> None:
    root = bitworld_root()
    table = Table(title="BitWorld Games", show_header=True)
    table.add_column("Game", style="cyan")
    table.add_column("Entry")
    for game in discover_games(root):
        table.add_row(game.name, str(game.entrypoint.relative_to(root)))
    console.print(table)


@bitworld_app.command("quick-run", help="Compile and run a BitWorld game with local player clients.")
def quick_run_cmd(
    game: Annotated[str, typer.Argument(help="BitWorld game folder name.")],
    port: Annotated[int | None, typer.Argument(help="Port to bind. If omitted, BitWorld chooses one.")] = None,
    players: Annotated[int, typer.Option("--players", min=1, help="Number of local player clients.")] = 1,
    address: Annotated[str, typer.Option("--address", help="Server bind address.")] = DEFAULT_ADDRESS,
    save_replay: Annotated[Path | None, typer.Option("--save-replay", help="Path for BitWorld replay output.")] = None,
    nim: Annotated[str, typer.Option("--nim", help="Nim compiler executable.")] = "nim",
) -> None:
    root = bitworld_root()
    config = QuickRunConfig(game=game, port=port, players=players, address=address, save_replay=save_replay)
    _require_game(root, config.game)
    executable = compile_quick_run(root, nim)
    subprocess.run(quick_run_args(executable, config), cwd=root, check=True)
