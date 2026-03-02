from __future__ import annotations

import json
from typing import Literal, Optional, TypeGuard, cast

import httpx
import typer
from rich import box
from rich.table import Table

from cogames.cli.base import console, emit_json
from cogames.cli.client import LeaderboardEntry, ScorePoliciesLeaderboardEntry, TeamSummary, TournamentServerClient
from cogames.cli.leaderboard import _format_score, _format_timestamp
from cogames.cli.login import DEFAULT_COGAMES_SERVER
from cogames.cli.submit import DEFAULT_SUBMIT_SERVER

LeaderboardEntries = list[LeaderboardEntry] | list[ScorePoliciesLeaderboardEntry] | list[TeamSummary]
LeaderboardType = Literal["policy", "team", "score-policies"]
LEADERBOARD_TYPE_OPTION = typer.Option(
    "policy",
    "--type",
    "-t",
    help="Leaderboard type (policy, team, score-policies).",
)


def _get_client(login_server: str, server: str) -> TournamentServerClient:
    _ = login_server
    return TournamentServerClient(server_url=server)


def _http_error_detail(exc: httpx.HTTPStatusError) -> str | None:
    try:
        payload = exc.response.json()
    except ValueError:
        return None
    if isinstance(payload, dict):
        detail = payload.get("detail")
        if isinstance(detail, str):
            return detail
    return None


def _team_label(team: TeamSummary, fallback: str) -> str:
    if team.id is not None:
        return str(team.id)
    if team.name:
        return team.name
    return fallback


def _team_members(team: TeamSummary) -> str:
    if team.cogs:
        return ", ".join(f"{cog.position}:{cog.policy.name or '?'}:v{cog.policy.version or '?'}" for cog in team.cogs)
    return ", ".join(f"{idx}:{member.name or '?'}:v{member.version or '?'}" for idx, member in enumerate(team.members))


def _is_team_entries(entries: LeaderboardEntries) -> TypeGuard[list[TeamSummary]]:
    return bool(entries) and isinstance(entries[0], TeamSummary)


def _format_optional_int(value: int | None) -> str:
    return str(value) if value is not None else "—"


def _format_optional_bool(value: bool | None) -> str:
    if value is None:
        return "—"
    return "yes" if value else "no"


season_app = typer.Typer(
    help="Tournament season commands.",
    context_settings={"help_option_names": ["-h", "--help"]},
    no_args_is_help=True,
    rich_markup_mode="rich",
)


@season_app.command(name="list", help="List tournament seasons.")
def season_list(
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
        "-s",
        metavar="URL",
        help="Tournament server URL.",
        rich_help_panel="Server",
    ),
    json_output: bool = typer.Option(False, "--json", help="Print raw JSON.", rich_help_panel="Output"),
) -> None:
    client = _get_client(login_server, server)

    with client:
        seasons = client.get_seasons()

    if json_output:
        emit_json([s.model_dump(mode="json") for s in seasons])
        return

    if not seasons:
        console.print("[yellow]No seasons found.[/yellow]")
        return

    table = Table(title="Tournament Seasons", box=box.SIMPLE_HEAVY, show_lines=False, pad_edge=False)
    table.add_column("Season", style="bold cyan")
    table.add_column("Description", style="white")

    for season in seasons:
        table.add_row(season.name, season.summary)

    console.print(table)


@season_app.command(name="show", help="Show details for a season.")
def season_show(
    season_name: str = typer.Argument(..., metavar="SEASON", help="Season name."),
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
        "-s",
        metavar="URL",
        help="Tournament server URL.",
        rich_help_panel="Server",
    ),
    json_output: bool = typer.Option(False, "--json", help="Print raw JSON.", rich_help_panel="Output"),
) -> None:
    client = _get_client(login_server, server)

    try:
        with client:
            info = client.get_season(season_name)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            console.print(f"[red]Season '{season_name}' not found[/red]")
            raise typer.Exit(1) from exc
        raise

    if json_output:
        emit_json(info.model_dump(mode="json"))
        return

    console.print(f"\n[bold]{info.display_name}[/bold]")
    console.print(f"Name: {info.name}")
    console.print(f"Status: {info.status}")
    console.print(f"Type: {info.tournament_type}")
    console.print(f"Version: v{info.version}")
    if info.compat_version:
        console.print(f"Compat: {info.compat_version}")
    if info.started_at:
        console.print(f"Started: {_format_timestamp(info.started_at)}")
    console.print(f"Entrants: {info.active_entrant_count} active / {info.entrant_count} total")
    console.print(f"Matches: {info.match_count}")
    console.print(f"Stages: {info.stage_count}")
    if info.pools:
        console.print("\n[bold]Pools:[/bold]")
        for pool in info.pools:
            console.print(f"  {pool.name}: {pool.description}")


@season_app.command(name="versions", help="List versions of a season.")
def season_versions(
    season_name: str = typer.Argument(..., metavar="SEASON", help="Season name."),
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
        "-s",
        metavar="URL",
        help="Tournament server URL.",
        rich_help_panel="Server",
    ),
    json_output: bool = typer.Option(False, "--json", help="Print raw JSON.", rich_help_panel="Output"),
) -> None:
    client = _get_client(login_server, server)

    with client:
        versions = client.get_season_versions(season_name)

    if json_output:
        emit_json([v.model_dump(mode="json") for v in versions])
        return

    if not versions:
        console.print(f"[yellow]No versions found for season '{season_name}'.[/yellow]")
        return

    table = Table(title=f"Versions: {season_name}", box=box.SIMPLE_HEAVY, show_lines=False, pad_edge=False)
    table.add_column("Version", style="bold cyan")
    table.add_column("Compat", style="white")
    table.add_column("Status", style="white")
    table.add_column("Created", style="dim")

    for v in versions:
        status = "[green]canonical[/green]" if v.canonical else "[dim]historical[/dim]"
        table.add_row(f"v{v.version}", v.compat_version or "—", status, _format_timestamp(v.created_at))

    console.print(table)


@season_app.command(name="stages", help="Show stages for a season.")
def season_stages(
    season_name: str = typer.Argument(..., metavar="SEASON", help="Season name."),
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
        "-s",
        metavar="URL",
        help="Tournament server URL.",
        rich_help_panel="Server",
    ),
    json_output: bool = typer.Option(False, "--json", help="Print raw JSON.", rich_help_panel="Output"),
) -> None:
    client = _get_client(login_server, server)

    try:
        with client:
            stages = client.get_stages(season_name)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            console.print(f"[red]Season '{season_name}' not found[/red]")
            raise typer.Exit(1) from exc
        raise

    if json_output:
        emit_json([s.model_dump(mode="json") for s in stages])
        return

    if not stages:
        console.print(f"[yellow]No stages found for season '{season_name}'.[/yellow]")
        return

    table = Table(title=f"Stages: {season_name}", box=box.SIMPLE_HEAVY, show_lines=False, pad_edge=False)
    table.add_column("Stage", style="bold cyan")
    table.add_column("Policies", justify="right")
    table.add_column("Matches", justify="right")
    table.add_column("Complete", justify="right")
    table.add_column("Teams", justify="right")
    table.add_column("Eliminated", justify="right")

    for stage in stages:
        table.add_row(
            stage.name,
            str(stage.policy_count),
            str(stage.match_count),
            f"{stage.completion_pct:.1f}%",
            str(stage.team_count) if stage.team_count is not None else "—",
            str(stage.eliminated_count) if stage.eliminated_count is not None else "—",
        )

    console.print(table)


@season_app.command(name="progress", help="Show season progress summary.")
def season_progress(
    season_name: str = typer.Argument(..., metavar="SEASON", help="Season name."),
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
        "-s",
        metavar="URL",
        help="Tournament server URL.",
        rich_help_panel="Server",
    ),
    json_output: bool = typer.Option(False, "--json", help="Print raw JSON.", rich_help_panel="Output"),
) -> None:
    client = _get_client(login_server, server)

    try:
        with client:
            progress = client.get_progress(season_name)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            console.print(f"[red]Season '{season_name}' not found[/red]")
            raise typer.Exit(1) from exc
        if exc.response.status_code == 400:
            detail = _http_error_detail(exc) or "Season progress is not available."
            console.print(f"[red]{detail}[/red]")
            raise typer.Exit(1) from exc
        raise

    if json_output:
        emit_json(progress.model_dump(mode="json"))
        return

    console.print(f"\n[bold]Progress: {season_name}[/bold]")
    console.print(f"Started: {'yes' if progress.started else 'no'}")
    console.print(f"Phase: {progress.phase}")
    if progress.phase_detail:
        detail = ", ".join(f"{key}={value}" for key, value in progress.phase_detail.items())
        console.print(f"Phase detail: {detail}")

    completed_stages = sum(1 for stage in progress.stage_flow if stage.status == "complete")
    console.print(f"Stages: {completed_stages}/{len(progress.stage_flow)}")
    active_stage = next((stage.name for stage in progress.stage_flow if stage.status == "active"), None)
    if active_stage:
        console.print(f"Current stage: {active_stage}")


@season_app.command(name="leaderboard", help="Show season leaderboard.")
def season_leaderboard(
    season_name: Optional[str] = typer.Argument(None, metavar="SEASON", help="Season name (default: server default)."),
    pool: Optional[str] = typer.Option(None, "--pool", "-p", help="Pool name for stage-specific leaderboard."),
    leaderboard_type: LeaderboardType = LEADERBOARD_TYPE_OPTION,
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
        "-s",
        metavar="URL",
        help="Tournament server URL.",
        rich_help_panel="Server",
    ),
    json_output: bool = typer.Option(False, "--json", help="Print raw JSON.", rich_help_panel="Output"),
) -> None:
    season = season_name or "<default>"
    client = _get_client(login_server, server)
    if leaderboard_type == "team" and pool is None:
        console.print("[red]Team leaderboard requires --pool <POOL>.[/red]")
        raise typer.Exit(1)

    try:
        with client:
            season = season_name or client.get_default_season().name
            if leaderboard_type == "score-policies" and pool is None:
                entries = client.get_score_policies_leaderboard(season)
            elif pool:
                entries = client.get_stage_leaderboard(season, leaderboard_type, pool)
            else:
                entries = client.get_leaderboard(season)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            console.print(f"[red]Season '{season}' not found[/red]")
            raise typer.Exit(1) from exc
        if exc.response.status_code == 400:
            detail = _http_error_detail(exc) or "Leaderboard request was invalid."
            console.print(f"[red]{detail}[/red]")
            raise typer.Exit(1) from exc
        raise

    if json_output:
        emit_json([e.model_dump(mode="json") for e in entries])
        return

    if not entries:
        title = f"{season} / {pool}" if pool else season
        console.print(f"[yellow]No leaderboard entries for '{title}'.[/yellow]")
        return

    if _is_team_entries(entries):
        title = f"Team Leaderboard: {season}"
        if pool:
            title += f" ({pool})"
        table = Table(title=title, box=box.SIMPLE_HEAVY, show_lines=False, pad_edge=False)
        table.add_column("Rank", justify="right", style="bold")
        table.add_column("Team", style="bold cyan")
        table.add_column("Pool", style="white")
        table.add_column("Score", justify="right", style="green")
        table.add_column("Matches", justify="right")
        table.add_column("Eliminated", justify="center")
        table.add_column("Members", style="dim")

        for idx, team in enumerate(entries, start=1):
            table.add_row(
                str(idx),
                _team_label(team, str(idx)),
                team.pool_name,
                _format_score(team.score),
                _format_optional_int(team.matches),
                _format_optional_bool(team.eliminated),
                _team_members(team),
            )

        console.print(table)
        return

    title = f"Leaderboard: {season}"
    if pool:
        title += f" ({pool})"
    table = Table(title=title, box=box.SIMPLE_HEAVY, show_lines=False, pad_edge=False)
    table.add_column("Rank", justify="right", style="bold")
    table.add_column("Policy", style="bold cyan")
    table.add_column("Score", justify="right", style="green")
    table.add_column("Matches", justify="right")

    non_team_entries = cast(list[LeaderboardEntry] | list[ScorePoliciesLeaderboardEntry], entries)
    for entry in non_team_entries:
        policy_label = f"{entry.policy.name or '?'}[dim]:v{entry.policy.version or '?'}[/dim]"
        if isinstance(entry, ScorePoliciesLeaderboardEntry):
            score = _format_score(entry.placement_score)
            matches = str(entry.team_appearances)
        else:
            score = _format_score(entry.score)
            matches = str(entry.matches)
        table.add_row(str(entry.rank), policy_label, score, matches)

    console.print(table)


@season_app.command(name="teams", help="Show teams in a season.")
def season_teams(
    season_name: str = typer.Argument(..., metavar="SEASON", help="Season name."),
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
        "-s",
        metavar="URL",
        help="Tournament server URL.",
        rich_help_panel="Server",
    ),
    json_output: bool = typer.Option(False, "--json", help="Print raw JSON.", rich_help_panel="Output"),
) -> None:
    client = _get_client(login_server, server)

    try:
        with client:
            teams = client.get_teams(season_name)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            console.print(f"[red]Season '{season_name}' not found[/red]")
            raise typer.Exit(1) from exc
        raise

    if json_output:
        emit_json([t.model_dump(mode="json") for t in teams])
        return

    if not teams:
        console.print(f"[yellow]No teams found for season '{season_name}'.[/yellow]")
        return

    table = Table(title=f"Teams: {season_name}", box=box.SIMPLE_HEAVY, show_lines=False, pad_edge=False)
    table.add_column("Team", style="bold cyan")
    table.add_column("Pool", style="white")
    table.add_column("Score", justify="right", style="green")
    table.add_column("Matches", justify="right")
    table.add_column("Eliminated", justify="center")
    table.add_column("Members", style="dim")

    for idx, team in enumerate(teams, start=1):
        table.add_row(
            _team_label(team, str(idx)),
            team.pool_name,
            _format_score(team.score),
            _format_optional_int(team.matches),
            _format_optional_bool(team.eliminated),
            _team_members(team),
        )

    console.print(table)


@season_app.command(name="matches", help="Show matches in a season.")
def season_matches(
    season_name: str = typer.Argument(..., metavar="SEASON", help="Season name."),
    limit: int = typer.Option(20, "--limit", "-n", help="Number of matches to show."),
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
        "-s",
        metavar="URL",
        help="Tournament server URL.",
        rich_help_panel="Server",
    ),
    json_output: bool = typer.Option(False, "--json", help="Print raw JSON.", rich_help_panel="Output"),
) -> None:
    client = _get_client(login_server, server)

    try:
        with client:
            matches = client.get_season_matches(season_name, limit=limit)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            console.print(f"[red]Season '{season_name}' not found[/red]")
            raise typer.Exit(1) from exc
        raise

    if json_output:
        emit_json([m.model_dump(mode="json") for m in matches])
        return

    if not matches:
        console.print(f"[yellow]No matches found in season '{season_name}'.[/yellow]")
        return

    table = Table(title=f"Matches: {season_name}", box=box.SIMPLE_HEAVY, show_lines=False, pad_edge=False)
    table.add_column("Match ID", style="dim", overflow="fold")
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
            str(match.id),
            _format_timestamp(match.created_at.isoformat()),
            f"[{status_style}]{match.status}[/{status_style}]",
            match.pool_name,
            policies,
            scores,
        )

    console.print(table)


@season_app.command(name="pool-config", help="Show configuration for a season pool.")
def season_pool_config(
    season_name: str = typer.Argument(..., metavar="SEASON", help="Season name."),
    pool_name: str = typer.Argument(..., metavar="POOL", help="Pool name."),
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
        "-s",
        metavar="URL",
        help="Tournament server URL.",
        rich_help_panel="Server",
    ),
    json_output: bool = typer.Option(False, "--json", help="Print raw JSON.", rich_help_panel="Output"),
) -> None:
    client = _get_client(login_server, server)

    try:
        with client:
            config = client.get_pool_config(season_name, pool_name)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            console.print(f"[red]Pool '{pool_name}' in season '{season_name}' not found[/red]")
            raise typer.Exit(1) from exc
        raise

    if isinstance(config, dict):
        config_payload = {"pool_name": pool_name, "config": config}
    else:
        config_payload = config.model_dump(mode="json")

    if json_output:
        emit_json(config_payload)
        return

    console.print(f"\n[bold]Pool Config: {config_payload['pool_name']}[/bold]")
    console.print(json.dumps(config_payload["config"], indent=2))
