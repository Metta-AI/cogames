"""CLI commands for viewing matches and policy logs."""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any, Optional

import httpx
import typer
from rich import box
from rich.table import Table

from cogames.cli.base import console
from cogames.cli.client import MatchResponse, TournamentServerClient
from cogames.cli.leaderboard import _format_score, _format_timestamp, parse_season_ref
from cogames.cli.login import DEFAULT_COGAMES_SERVER
from cogames.cli.submit import DEFAULT_SUBMIT_SERVER


def _get_client(login_server: str, server: str) -> TournamentServerClient | None:
    return TournamentServerClient.from_login(server_url=server, login_server=login_server)


def _help_callback(ctx: typer.Context, value: bool) -> None:
    """Callback for custom help option."""
    if value:
        console.print(ctx.get_help())
        raise typer.Exit()


def matches_cmd(
    match_id: Optional[str] = typer.Argument(
        None,
        metavar="MATCH_ID",
        help="Match ID to show details for. If omitted, lists recent matches.",
    ),
    season: Optional[str] = typer.Option(
        None,
        "--season",
        "-s",
        metavar="SEASON",
        help="Tournament season (for listing matches).",
        rich_help_panel="Filter",
    ),
    limit: int = typer.Option(
        20,
        "--limit",
        "-n",
        metavar="N",
        help="Number of matches to show.",
        rich_help_panel="Filter",
    ),
    logs: bool = typer.Option(
        False,
        "--logs",
        "-l",
        help="Show available policy logs for the match.",
        rich_help_panel="Output",
    ),
    download_logs: Optional[Path] = typer.Option(  # noqa: B008
        None,
        "--download-logs",
        "-d",
        metavar="DIR",
        help="Download all accessible logs to directory.",
        rich_help_panel="Output",
    ),
    login_server: str = typer.Option(
        DEFAULT_COGAMES_SERVER,
        "--login-server",
        metavar="URL",
        help="Authentication server URL.",
        rich_help_panel="Server",
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
    client = _get_client(login_server, server)
    if not client:
        return

    with client:
        if match_id:
            _show_match_detail(client, match_id, logs, download_logs, json_output)
        else:
            _list_matches(client, season, limit, json_output)


def _list_matches(
    client: TournamentServerClient,
    season: Optional[str],
    limit: int,
    json_output: bool,
) -> None:
    """List recent matches for the user's policies."""
    try:
        # Get user's policy versions
        my_policies = client.get_my_policy_versions()
        if not my_policies:
            console.print("[yellow]No uploaded policies found.[/yellow]")
            console.print("Upload a policy with: [cyan]cogames upload[/cyan]")
            return

        policy_version_ids = [pv.id for pv in my_policies]

        # Determine season
        if season is None:
            default_season = client.get_default_season()
            season = default_season.name

        season_name, _ = parse_season_ref(season)

        # Fetch matches
        matches = client.get_season_matches(
            season_name,
            include_hidden_seasons=True,
            policy_version_ids=policy_version_ids,
            limit=limit,
        )

    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            console.print(f"[red]Season '{season}' not found[/red]")
            raise typer.Exit(1) from exc
        console.print(f"[red]Request failed with status {exc.response.status_code}[/red]")
        raise typer.Exit(1) from exc
    except httpx.HTTPError as exc:
        console.print(f"[red]Failed to reach server:[/red] {exc}")
        raise typer.Exit(1) from exc

    if json_output:
        console.print(json.dumps([m.model_dump(mode="json") for m in matches], indent=2))
        return

    if not matches:
        console.print(f"[yellow]No matches found in season '{season}'.[/yellow]")
        return

    table = Table(
        title=f"Recent Matches ({season})",
        box=box.SIMPLE_HEAVY,
        show_lines=False,
        pad_edge=False,
    )
    table.add_column("Match ID", style="dim")
    table.add_column("Date", style="green")
    table.add_column("Status")
    table.add_column("Pool")
    table.add_column("Policies", style="cyan")
    table.add_column("Score", justify="right")

    for match in matches:
        status_style = {
            "completed": "green",
            "failed": "red",
            "running": "yellow",
            "pending": "dim",
        }.get(match.status, "white")

        policies = ", ".join(f"{p.policy.name or '?'}:v{p.policy.version or '?'}" for p in match.players)
        scores = ", ".join(_format_score(p.score) for p in match.players)

        table.add_row(
            str(match.id)[:8],
            _format_timestamp(match.created_at.isoformat()),
            f"[{status_style}]{match.status}[/{status_style}]",
            match.pool_name,
            policies,
            scores,
        )

    console.print(table)
    console.print("\n[dim]View details: cogames matches <match-id>[/dim]")
    console.print("[dim]View logs: cogames matches <match-id> --logs[/dim]")


def _resolve_match_id(client: TournamentServerClient, match_id_str: str) -> uuid.UUID:
    """Resolve a full or prefix match ID to a UUID.

    Accepts a full UUID or a short prefix (as shown by ``cogames matches``).
    """
    try:
        return uuid.UUID(match_id_str)
    except ValueError:
        pass

    # Treat as a prefix — search recent matches for a unique hit
    my_policies = client.get_my_policy_versions()
    if not my_policies:
        console.print(f"[red]Invalid match ID: {match_id_str}[/red]")
        raise typer.Exit(1)

    default_season = client.get_default_season()
    matches = client.get_season_matches(
        default_season.name,
        include_hidden_seasons=True,
        policy_version_ids=[pv.id for pv in my_policies],
    )
    hits = [m for m in matches if str(m.id).startswith(match_id_str)]
    if len(hits) == 1:
        return hits[0].id
    if len(hits) > 1:
        console.print(f"[red]Ambiguous match ID prefix '{match_id_str}' — matches {len(hits)} matches[/red]")
        raise typer.Exit(1)

    console.print(f"[red]No match found with prefix '{match_id_str}'[/red]")
    raise typer.Exit(1)


def _show_match_detail(
    client: TournamentServerClient,
    match_id_str: str,
    show_logs: bool,
    download_logs: Optional[Path],
    json_output: bool,
) -> None:
    """Show details for a specific match."""
    match_uuid = _resolve_match_id(client, match_id_str)

    try:
        match = client.get_match(match_uuid)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            console.print(f"[red]Match '{match_id_str}' not found[/red]")
            raise typer.Exit(1) from exc
        console.print(f"[red]Request failed with status {exc.response.status_code}[/red]")
        raise typer.Exit(1) from exc
    except httpx.HTTPError as exc:
        console.print(f"[red]Failed to reach server:[/red] {exc}")
        raise typer.Exit(1) from exc

    if json_output:
        result: dict[str, Any] = match.model_dump(mode="json")
        if show_logs:
            result["logs"] = _collect_logs(client, match)
        console.print(json.dumps(result, indent=2))
        return

    # Display match info
    console.print(f"\n[bold]Match {match.id}[/bold]")
    console.print(f"Season: {match.season_name}")
    console.print(f"Pool: {match.pool_name}")
    console.print(f"Status: {match.status}")
    console.print(f"Date: {_format_timestamp(match.created_at.isoformat())}")

    if match.error:
        # Show only the last line of the traceback
        error_summary = match.error.strip().rsplit("\n", 1)[-1]
        console.print(f"[red]Error: {error_summary}[/red]")

    console.print("\n[bold]Players:[/bold]")
    for i, player in enumerate(match.players):
        policy_label = f"{player.policy.name or '?'}:v{player.policy.version or '?'}"
        score = _format_score(player.score)
        console.print(f"  {i}: {policy_label} ({player.num_agents} agents) - Score: {score}")

    # Handle logs
    if show_logs or download_logs:
        # Find which policy versions the user owns in this match
        my_policies = client.get_my_policy_versions()
        my_pv_ids = {pv.id for pv in my_policies}
        match_pv_ids = [p.policy.id for p in match.players if p.policy.id in my_pv_ids]

        if not match_pv_ids:
            console.print("\n[yellow]You don't own any policies in this match.[/yellow]")
            return

        for pv_id in match_pv_ids:
            _handle_logs(client, match.id, pv_id, download_logs)


def _collect_logs(client: TournamentServerClient, match: MatchResponse) -> dict[str, list[str]]:
    """Collect log file lists for the user's policies in a match, keyed by policy_version_id."""
    my_policies = client.get_my_policy_versions()
    my_pv_ids = {pv.id for pv in my_policies}
    result: dict[str, list[str]] = {}
    for p in match.players:
        if p.policy.id not in my_pv_ids:
            continue
        try:
            files = client.list_match_policy_logs(match.id, p.policy.id)
        except httpx.HTTPStatusError:
            continue
        result[str(p.policy.id)] = files
    return result


def _handle_logs(
    client: TournamentServerClient,
    match_id: uuid.UUID,
    policy_version_id: uuid.UUID,
    download_dir: Optional[Path],
) -> None:
    """List or download policy logs for a policy in a match."""
    try:
        log_files = client.list_match_policy_logs(match_id, policy_version_id)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 403:
            console.print("\n[yellow]No logs accessible (you may not own any policies in this match).[/yellow]")
            return
        console.print(f"\n[red]Failed to fetch logs: {exc.response.status_code}[/red]")
        return
    except httpx.HTTPError as exc:
        console.print(f"\n[red]Failed to fetch logs:[/red] {exc}")
        return

    if not log_files:
        console.print("\n[yellow]No logs available for this match.[/yellow]")
        return

    console.print(f"\n[bold]Available Logs ({len(log_files)}):[/bold]")
    for f in log_files:
        console.print(f"  {f}")

    if download_dir:
        download_dir.mkdir(parents=True, exist_ok=True)
        console.print(f"\n[cyan]Downloading logs to {download_dir}...[/cyan]")

        for filename in log_files:
            fname_match = re.match(r"policy_agent_(\d+)\.txt", filename)
            if not fname_match:
                continue
            agent_idx = int(fname_match.group(1))

            try:
                content = client.get_match_policy_log(match_id, policy_version_id, agent_idx)
                out_path = download_dir / filename
                out_path.write_text(content)
                console.print(f"  [green]✓[/green] {filename}")
            except httpx.HTTPStatusError as exc:
                console.print(f"  [red]✗[/red] {filename}: {exc.response.status_code}")
            except httpx.HTTPError as exc:
                console.print(f"  [red]✗[/red] {filename}: {exc}")

        console.print(f"\n[green]Logs saved to {download_dir}[/green]")
    else:
        console.print(f"\n[dim]Download logs: cogames matches {match_id} --download-logs ./logs[/dim]")
