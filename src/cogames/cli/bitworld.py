"""BitWorld CLI bridge for installed websocket games."""

from __future__ import annotations

import http.server
import importlib.resources
import importlib.util
import os
import socket
import subprocess
import threading
import time
import webbrowser
from functools import partial
from pathlib import Path
from typing import Annotated

import typer
from pydantic import BaseModel, ConfigDict, Field
from rich.table import Table

from cogames.cli.base import console

BITWORLD_EXTRA = "bitworld"
BITWORLD_ROOT_ENV = "COGAMES_BITWORLD_ROOT"
BITREPLAY_MAGIC = b"BITWORLD"
BITREPLAY_HEADER_PREFIX_BYTES = len(BITREPLAY_MAGIC) + 4
DEFAULT_ADDRESS = "127.0.0.1"
DEFAULT_BROWSER_ADDRESS = "localhost"
DEFAULT_CLIENT_SERVER_ADDRESS = "127.0.0.1"
PORT_POLL_SECONDS = 0.1
PORT_WAIT_SECONDS = 8.0
STOP_WAIT_SECONDS = 2.0
SKIP_FOLDERS = frozenset(
    {
        "__pycache__",
        "client",
        "clients",
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


class ReplayConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    replay_path: Path
    game: str | None = Field(default=None, min_length=1)
    port: int | None = Field(default=None, ge=1, le=65535)
    address: str = Field(default=DEFAULT_ADDRESS, min_length=1)
    browser_address: str = Field(default=DEFAULT_BROWSER_ADDRESS, min_length=1)
    duration: float | None = Field(default=None, ge=0)


bitworld_app = typer.Typer(
    help="Run optional BitWorld websocket games.",
    context_settings={"help_option_names": ["-h", "--help"]},
    no_args_is_help=True,
    rich_markup_mode="rich",
)


def bitworld_root() -> Path:
    env_root = os.environ.get(BITWORLD_ROOT_ENV)
    if env_root:
        return Path(env_root).expanduser().resolve()

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


def _binary_path(source: Path) -> Path:
    suffix = ".exe" if os.name == "nt" else ""
    return source.with_suffix(suffix)


def _choose_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind(("127.0.0.1", 0))
        return int(server_socket.getsockname()[1])


def _wait_for_port(host: str, port: int, process: subprocess.Popen) -> None:
    addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    deadline = time.monotonic() + PORT_WAIT_SECONDS
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"BitWorld replay server exited before listening on {host}:{port}")
        for family, socket_type, proto, _canonname, sockaddr in addresses:
            with socket.socket(family, socket_type, proto) as probe:
                probe.settimeout(PORT_POLL_SECONDS)
                if probe.connect_ex(sockaddr) == 0:
                    return
        time.sleep(PORT_POLL_SECONDS)
    raise RuntimeError(f"BitWorld replay server did not accept connections on {host}:{port}")


def _stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    deadline = time.monotonic() + STOP_WAIT_SECONDS
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(PORT_POLL_SECONDS)
    if process.poll() is None:
        process.kill()
    process.wait()


class _QuietClientHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return


class BitWorldClientServer:
    def __init__(self, clients_dir: Path, address: str = DEFAULT_CLIENT_SERVER_ADDRESS) -> None:
        self._server = http.server.ThreadingHTTPServer(
            (address, 0),
            partial(_QuietClientHandler, directory=str(clients_dir)),
        )
        self._address = address
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def port(self) -> int:
        return int(self._server.server_address[1])

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=STOP_WAIT_SECONDS)

    def global_client_url(self, browser_address: str, game_port: int) -> str:
        return (
            f"http://{self._address}:{self.port}/global_client.html?address=ws://{browser_address}:{game_port}/global"
        )


def start_client_server(root: Path) -> BitWorldClientServer:
    client_server = BitWorldClientServer(root / "clients")
    client_server.start()
    return client_server


def compile_quick_run(root: Path, nim: str) -> Path:
    subprocess.run([nim, "c", "tools/quick_run.nim"], cwd=root, check=True)
    return _quick_run_binary(root)


def compile_game(root: Path, game: BitWorldGame, nim: str) -> Path:
    source_relative = game.entrypoint.relative_to(root).as_posix()
    subprocess.run([nim, "c", source_relative], cwd=root, check=True)
    return _binary_path(game.entrypoint)


def quick_run_args(executable: Path, config: QuickRunConfig) -> list[str]:
    args = [str(executable), config.game]
    if config.port is not None:
        args.append(str(config.port))
    args.append(f"--players:{config.players}")
    args.append(f"--address:{config.address}")
    if config.save_replay is not None:
        args.append(f"--save-replay:{config.save_replay}")
    return args


def bitworld_replay_game_from_file(replay_path: Path) -> str:
    with replay_path.open("rb") as replay_file:
        prefix = replay_file.read(BITREPLAY_HEADER_PREFIX_BYTES)
        game_name_size = int.from_bytes(prefix[-2:], "little")
        return replay_file.read(game_name_size).decode("utf-8")


def launch_bitworld_replay(config: ReplayConfig, nim: str = "nim") -> int:
    root = bitworld_root()
    game = _require_game(root, config.game or bitworld_replay_game_from_file(config.replay_path))
    executable = compile_game(root, game, nim)
    port = config.port or _choose_port()

    process = subprocess.Popen(
        [
            str(executable),
            f"--address:{config.address}",
            f"--port:{port}",
            f"--load-replay:{config.replay_path.expanduser().resolve()}",
        ],
        cwd=game.entrypoint.parent,
    )
    try:
        _wait_for_port(config.address, port, process)
        client_server = start_client_server(root)
        try:
            url = client_server.global_client_url(config.browser_address, port)
            console.print(f"[cyan]Launching BitWorld replay viewer: {url}[/cyan]")
            webbrowser.open(url)
            if config.duration is not None:
                time.sleep(config.duration)
                return 0
            return int(process.wait())
        finally:
            client_server.stop()
    finally:
        _stop_process(process)


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


@bitworld_app.command("replay", help="Replay a BitWorld recording in the global web viewer.")
def replay_cmd(
    replay_path: Annotated[Path, typer.Argument(help="BitWorld replay file.")],
    game: Annotated[
        str | None,
        typer.Option("--game", help="BitWorld game folder override. Defaults to the replay header game."),
    ] = None,
    port: Annotated[int | None, typer.Option("--port", help="Port to bind. If omitted, one is chosen.")] = None,
    address: Annotated[str, typer.Option("--address", help="Server bind address.")] = DEFAULT_ADDRESS,
    browser_address: Annotated[
        str,
        typer.Option("--browser-address", help="Address the browser should use to reach the replay server."),
    ] = DEFAULT_BROWSER_ADDRESS,
    duration: Annotated[
        float | None,
        typer.Option("--duration", help="Seconds to keep the replay server alive. If omitted, wait until interrupted."),
    ] = None,
    nim: Annotated[str, typer.Option("--nim", help="Nim compiler executable.")] = "nim",
) -> None:
    exit_code = launch_bitworld_replay(
        ReplayConfig(
            replay_path=replay_path,
            game=game,
            port=port,
            address=address,
            browser_address=browser_address,
            duration=duration,
        ),
        nim=nim,
    )
    if exit_code != 0:
        raise typer.Exit(exit_code)
