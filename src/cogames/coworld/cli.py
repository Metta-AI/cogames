from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from cogames.cli.submit import DEFAULT_SUBMIT_SERVER
from cogames.coworld.certifier import certify_coworld
from cogames.coworld.manifest_uri import materialized_manifest_path
from cogames.coworld.play import PlaySession, ReplaySession, play_coworld, replay_coworld
from cogames.coworld.runner.runner import EpisodeArtifacts, run_coworld_episode
from cogames.coworld.types import CoworldEpisodeJobSpec
from cogames.coworld.upload import upload_coworld_cmd, upload_policy_cmd
from softmax.auth import DEFAULT_COGAMES_SERVER

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)


@app.callback()
def main() -> None:
    """Validate, certify, and play Coworld packages."""


@app.command("certify")
def certify(
    manifest_path: Annotated[Path, typer.Argument(help="Path to coworld_manifest.json.")],
    timeout_seconds: Annotated[float, typer.Option("--timeout-seconds", min=1.0)] = 60.0,
) -> None:
    result = certify_coworld(manifest_path, timeout_seconds=timeout_seconds)
    typer.echo(f"Certified {manifest_path}")
    typer.echo(f"Artifacts: {result.artifacts.workspace}")
    typer.echo(f"Results: {result.artifacts.results_path}")
    typer.echo(f"Replay: {result.artifacts.replay_path}")
    typer.echo(f"Logs: {result.artifacts.logs_dir}")


@app.command("play")
def play(
    manifest_uri: Annotated[str, typer.Argument(help="Path or URI to coworld_manifest.json.")],
    timeout_seconds: Annotated[float, typer.Option("--timeout-seconds", min=1.0, help="Health check timeout.")] = 60.0,
) -> None:
    with materialized_manifest_path(manifest_uri) as manifest_path:
        result = play_coworld(manifest_path, timeout_seconds=timeout_seconds, on_ready=_print_play_session)
    typer.echo(f"Results: {result.session.artifacts.results_path}")
    typer.echo(f"Replay: {result.session.artifacts.replay_path}")
    typer.echo(f"Logs: {result.session.artifacts.logs_dir}")


@app.command("upload-coworld")
def upload_coworld(
    manifest_path: Annotated[Path, typer.Argument(help="Path to coworld_manifest.json.")],
    server: Annotated[str, typer.Option("--server", help="Observatory API server URL.")] = DEFAULT_SUBMIT_SERVER,
    login_server: Annotated[
        str,
        typer.Option("--login-server", help="Authentication server URL."),
    ] = DEFAULT_COGAMES_SERVER,
    timeout_seconds: Annotated[float, typer.Option("--timeout-seconds", min=1.0)] = 60.0,
) -> None:
    upload_coworld_cmd(
        manifest_path,
        server=server,
        login_server=login_server,
        timeout_seconds=timeout_seconds,
    )


@app.command("upload-policy")
def upload_policy(
    image: Annotated[str, typer.Argument(help="Local Docker image to upload as a CoWorld policy.")],
    name: Annotated[str, typer.Option("--name", "-n", help="Policy name.")],
    run: Annotated[
        list[str] | None,
        typer.Option("--run", help="Command argv for images that contain multiple Coworld roles."),
    ] = None,
    server: Annotated[str, typer.Option("--server", help="Observatory API server URL.")] = DEFAULT_SUBMIT_SERVER,
    login_server: Annotated[
        str,
        typer.Option("--login-server", help="Authentication server URL."),
    ] = DEFAULT_COGAMES_SERVER,
) -> None:
    upload_policy_cmd(
        image,
        name,
        run=run,
        server=server,
        login_server=login_server,
    )


@app.command("run-episode")
def run_episode(
    spec_path: Annotated[Path, typer.Argument(help="Path to a CoworldEpisodeJobSpec JSON file.")],
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o", help="Directory for episode artifacts.")] = Path(
        "./coworld-episode-results"
    ),
    timeout_seconds: Annotated[float, typer.Option("--timeout-seconds", min=1.0)] = 3600.0,
    verify_replay: Annotated[bool, typer.Option("--verify-replay/--no-verify-replay")] = False,
) -> None:
    spec = CoworldEpisodeJobSpec.model_validate_json(spec_path.read_text(encoding="utf-8"))
    artifacts = EpisodeArtifacts.create(output_dir.resolve(), prefix="coworld-run-")
    run_coworld_episode(spec, artifacts, timeout_seconds=timeout_seconds, verify_replay=verify_replay)
    typer.echo(f"Artifacts: {artifacts.workspace}")
    typer.echo(f"Results: {artifacts.results_path}")
    typer.echo(f"Replay: {artifacts.replay_path}")
    typer.echo(f"Logs: {artifacts.logs_dir}")


@app.command("replay")
def replay(
    manifest_uri: Annotated[str, typer.Argument(help="Path or URI to coworld_manifest.json.")],
    replay_path: Annotated[Path, typer.Argument(help="Path to a replay artifact JSON file.")],
    timeout_seconds: Annotated[float, typer.Option("--timeout-seconds", min=1.0, help="Health check timeout.")] = 60.0,
) -> None:
    with materialized_manifest_path(manifest_uri) as manifest_path:
        session = replay_coworld(
            manifest_path,
            replay_path,
            timeout_seconds=timeout_seconds,
            on_ready=_print_replay_session,
        )
    typer.echo(f"Logs: {session.artifacts.logs_dir}")


def _print_play_session(session: PlaySession) -> None:
    typer.echo(f"Artifacts: {session.artifacts.workspace}")
    typer.echo("Player clients:")
    for slot, link in enumerate(session.links.players):
        typer.echo(f"  {slot}: {link}")
    typer.echo(f"Global client: {session.links.global_}")
    typer.echo(f"Admin client: {session.links.admin}")
    typer.echo("Waiting for the game container to exit...")


def _print_replay_session(session: ReplaySession) -> None:
    typer.echo(f"Artifacts: {session.artifacts.workspace}")
    typer.echo(f"Replay file: {session.replay_path}")
    typer.echo(f"Replay client: {session.link}")
    typer.echo("Waiting for the replay container to exit...")
