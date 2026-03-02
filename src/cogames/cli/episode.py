from __future__ import annotations

import importlib.util
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Optional

import httpx
import typer
from rich import box
from rich.table import Table

from cogames.cli.base import console, emit_json
from cogames.cli.client import EpisodeResponse, TournamentServerClient
from cogames.cli.leaderboard import _format_score, _format_timestamp
from cogames.cli.submit import DEFAULT_SUBMIT_SERVER

episode_app = typer.Typer(
    name="episode",
    help="Browse and replay game episodes.",
    rich_markup_mode="rich",
)


def _help_callback(ctx: typer.Context, value: bool) -> None:
    if value:
        console.print(ctx.get_help())
        raise typer.Exit()


def _get_anon_client(server: str) -> TournamentServerClient:
    return TournamentServerClient(server_url=server)


def _get_auth_client(login_server: str, server: str) -> TournamentServerClient | None:
    return TournamentServerClient.from_login(server_url=server, login_server=login_server)


def _resolve_episode_id(episode_id_str: str) -> uuid.UUID:
    try:
        return uuid.UUID(episode_id_str)
    except ValueError as exc:
        console.print(f"[red]Invalid episode ID: {episode_id_str}[/red]")
        raise typer.Exit(1) from exc


def _print_episode_table(episodes: list[EpisodeResponse]) -> None:
    table = Table(
        title="Episodes",
        box=box.SIMPLE_HEAVY,
        show_lines=False,
        pad_edge=False,
    )
    table.add_column("Episode ID", style="dim", overflow="fold")
    table.add_column("Date", style="green")
    table.add_column("Steps", justify="right")
    table.add_column("Policies", style="cyan")
    table.add_column("Avg Reward", justify="right")
    table.add_column("Replay", style="dim")

    for ep in episodes:
        policies = ", ".join(f"{pr.policy.name or '?'}:v{pr.policy.version or '?'}" for pr in ep.policy_results)
        rewards = ", ".join(_format_score(pr.avg_reward) for pr in ep.policy_results)
        replay_indicator = "yes" if ep.replay_url else "—"
        table.add_row(
            str(ep.id),
            _format_timestamp(ep.created_at.isoformat()),
            str(ep.steps) if ep.steps is not None else "—",
            policies,
            rewards,
            replay_indicator,
        )

    console.print(table)


def _print_episode_detail(ep: EpisodeResponse) -> None:
    console.print(f"\n[bold]Episode {ep.id}[/bold]")
    console.print(f"Date: {_format_timestamp(ep.created_at.isoformat())}")
    console.print(f"Steps: {ep.steps or '—'}")

    if ep.tags:
        tag_str = ", ".join(f"{k}={v}" for k, v in sorted(ep.tags.items()))
        console.print(f"Tags: {tag_str}")

    if ep.replay_url:
        console.print(f"Replay: {ep.replay_url}")

    if ep.policy_results:
        console.print("\n[bold]Policy Results:[/bold]")
        for pr in ep.policy_results:
            label = f"{pr.policy.name or '?'}:v{pr.policy.version or '?'}"
            console.print(
                f"  #{pr.position}: {label} ({pr.num_agents} agents) - Avg Reward: {_format_score(pr.avg_reward)}"
            )

    if ep.game_stats:
        console.print("\n[bold]Game Stats:[/bold]")
        for k, v in sorted(ep.game_stats.items()):
            console.print(f"  {k}: {v:.4f}")


@episode_app.command(
    name="list",
    help="List game episodes.",
    add_help_option=False,
)
def episode_list_cmd(
    policy: Optional[str] = typer.Option(
        None,
        "--policy",
        "-p",
        metavar="POLICY_VERSION_ID",
        help="Filter by policy version ID.",
        rich_help_panel="Filter",
    ),
    tags: Optional[str] = typer.Option(
        None,
        "--tags",
        "-t",
        metavar="TAGS",
        help="Comma-separated key:value tag filters.",
        rich_help_panel="Filter",
    ),
    limit: int = typer.Option(
        20,
        "--limit",
        "-n",
        metavar="N",
        help="Number of episodes to show.",
        rich_help_panel="Filter",
    ),
    server: str = typer.Option(
        DEFAULT_SUBMIT_SERVER,
        "--server",
        metavar="URL",
        help="Tournament server URL.",
        rich_help_panel="Server",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Print raw JSON instead of table.",
        rich_help_panel="Output",
    ),
    _help: bool = typer.Option(
        False,
        "--help",
        "-h",
        help="Show this message and exit.",
        is_eager=True,
        callback=_help_callback,
        rich_help_panel="Other",
    ),
) -> None:
    policy_version_id = uuid.UUID(policy) if policy else None

    with _get_anon_client(server) as client:
        episodes = client.list_episodes(
            policy_version_id=policy_version_id,
            tags=tags,
            limit=limit,
        )

    if json_output:
        emit_json([e.model_dump(mode="json") for e in episodes])
        return

    if not episodes:
        console.print("[yellow]No episodes found.[/yellow]")
        return

    _print_episode_table(episodes)
    console.print("\n[dim]View details: cogames episode show <episode-id>[/dim]")


@episode_app.command(
    name="show",
    help="Show details for a specific episode.",
    add_help_option=False,
)
def episode_show_cmd(
    episode_id: str = typer.Argument(
        ...,
        metavar="EPISODE_ID",
        help="Episode ID to show.",
    ),
    server: str = typer.Option(
        DEFAULT_SUBMIT_SERVER,
        "--server",
        metavar="URL",
        help="Tournament server URL.",
        rich_help_panel="Server",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Print raw JSON instead of table.",
        rich_help_panel="Output",
    ),
    _help: bool = typer.Option(
        False,
        "--help",
        "-h",
        help="Show this message and exit.",
        is_eager=True,
        callback=_help_callback,
        rich_help_panel="Other",
    ),
) -> None:
    ep_uuid = _resolve_episode_id(episode_id)

    with _get_anon_client(server) as client:
        ep = client.get_episode(ep_uuid)

    if json_output:
        emit_json(ep.model_dump(mode="json"))
        return

    _print_episode_detail(ep)

    if ep.replay_url:
        console.print(f"\n[dim]Replay: cogames episode replay {episode_id}[/dim]")


@episode_app.command(
    name="replay",
    help="Download and replay a game episode in MettaScope.",
    add_help_option=False,
)
def episode_replay_cmd(
    episode_id: str = typer.Argument(
        ...,
        metavar="EPISODE_ID",
        help="Episode ID to replay.",
    ),
    output: Optional[Path] = typer.Option(  # noqa: B008
        None,
        "--output",
        "-o",
        metavar="FILE",
        help="Save replay to file instead of launching viewer.",
        rich_help_panel="Output",
    ),
    server: str = typer.Option(
        DEFAULT_SUBMIT_SERVER,
        "--server",
        metavar="URL",
        help="Tournament server URL.",
        rich_help_panel="Server",
    ),
    _help: bool = typer.Option(
        False,
        "--help",
        "-h",
        help="Show this message and exit.",
        is_eager=True,
        callback=_help_callback,
        rich_help_panel="Other",
    ),
) -> None:
    ep_uuid = _resolve_episode_id(episode_id)

    with _get_anon_client(server) as client:
        ep = client.get_episode(ep_uuid)

    if not ep.replay_url:
        console.print("[red]No replay available for this episode.[/red]")
        raise typer.Exit(1)

    console.print(f"[cyan]Downloading replay for episode {str(ep.id)[:8]}...[/cyan]")
    replay_data = httpx.get(ep.replay_url, follow_redirects=True)
    replay_data.raise_for_status()

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(replay_data.content)
        console.print(f"[green]Replay saved to {output}[/green]")
        return

    _launch_mettascope(replay_data.content, ep.id)


def _launch_mettascope(replay_bytes: bytes, episode_id: uuid.UUID) -> None:
    spec = importlib.util.find_spec("mettagrid")
    if spec is None or spec.origin is None:
        console.print("[red]mettagrid package not available; cannot locate MettaScope.[/red]")
        raise typer.Exit(1)

    package_dir = Path(spec.origin).resolve().parent
    mettascope_path: Path | None = None
    for root in (package_dir, *package_dir.parents):
        candidate = root / "nim" / "mettascope" / "src" / "mettascope.nim"
        if candidate.exists():
            mettascope_path = candidate
            break

    if mettascope_path is None:
        console.print("[red]MettaScope sources not found.[/red]")
        raise typer.Exit(1)

    with tempfile.NamedTemporaryFile(suffix=".replay", delete=False, prefix=f"episode-{str(episode_id)[:8]}-") as f:
        f.write(replay_bytes)
        replay_path = Path(f.name)

    console.print(f"[cyan]Launching MettaScope to replay episode {str(episode_id)[:8]}...[/cyan]")
    cmd = ["nim", "r", str(mettascope_path), f"--replay:{replay_path}"]
    subprocess.run(cmd, check=True)
