from __future__ import annotations

import typer
from rich.table import Table

from cogames.cli.base import cli_http_errors, console, emit_json
from cogames.cli.client import TournamentServerClient
from softmax.auth import (
    DEFAULT_COGAMES_API_SERVER,
    DEFAULT_COGAMES_SERVER,
    fetch_cogames_whoami,
    load_cogames_user_token,
    restore_cogames_user_session,
    save_cogames_active_token,
)

player_app = typer.Typer(
    help="Manage CoGames player sessions.",
    context_settings={"help_option_names": ["-h", "--help"]},
    no_args_is_help=True,
    rich_markup_mode="rich",
)


def _load_user_token_or_exit(*, login_server: str) -> str:
    user_token = load_cogames_user_token(login_server=login_server)
    if user_token is None:
        console.print("[red]No saved user session found.[/red] Run [cyan]softmax login[/cyan] first.")
        raise typer.Exit(1)
    return user_token


@player_app.command(name="list")
def list_players_cmd(
    server: str = typer.Option(
        DEFAULT_COGAMES_API_SERVER,
        "--server",
        "-s",
        metavar="URL",
        help="Tournament API server URL.",
    ),
    login_server: str = typer.Option(
        DEFAULT_COGAMES_SERVER,
        "--login-server",
        metavar="URL",
        help="Authentication server URL.",
    ),
    json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """List the players owned by the saved user session."""
    user_token = _load_user_token_or_exit(login_server=login_server)
    with cli_http_errors("players"):
        with TournamentServerClient(server_url=server, token=user_token, login_server=login_server) as client:
            players = client.list_players()

    if json:
        emit_json([player.model_dump(mode="json") for player in players])
        return

    if not players:
        console.print("[yellow]No players found.[/yellow]")
        return

    table = Table(show_header=True)
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Created")
    for player in players:
        table.add_row(player.id, player.name, player.created_at.isoformat())
    console.print(table)


@player_app.command(name="login")
def login_player_cmd(
    player_id: str = typer.Argument(..., metavar="PLAYER_ID", help="Player id to activate, such as ply_..."),
    server: str = typer.Option(
        DEFAULT_COGAMES_API_SERVER,
        "--server",
        "-s",
        metavar="URL",
        help="Tournament API server URL.",
    ),
    login_server: str = typer.Option(
        DEFAULT_COGAMES_SERVER,
        "--login-server",
        metavar="URL",
        help="Authentication server URL.",
    ),
    json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Mint a short-lived player session token and make it the active CoGames session."""
    user_token = _load_user_token_or_exit(login_server=login_server)
    with cli_http_errors("player"):
        with TournamentServerClient(server_url=server, token=user_token, login_server=login_server) as client:
            response = client.login_player(player_id)

    save_cogames_active_token(login_server=login_server, token=response.token)

    if json:
        emit_json(response.model_dump(mode="json"))
        return

    console.print(f"[green]Player session active:[/green] {response.player_id}")
    console.print("subject_type: player")
    console.print(f"subject_id: {response.player_id}")
    console.print(f"expires_at: {response.expires_at.isoformat()}")
    console.print("[dim]Run `softmax status` or `cogames auth status` to inspect the active session.[/dim]")


@player_app.command(name="logout")
def logout_player_cmd(
    login_server: str = typer.Option(
        DEFAULT_COGAMES_SERVER,
        "--login-server",
        metavar="URL",
        help="Authentication server URL.",
    ),
) -> None:
    """Restore the saved user session as the active CoGames session."""
    user_token = restore_cogames_user_session(login_server=login_server)
    if user_token is None:
        console.print("[red]No saved user session found.[/red] Run [cyan]softmax login[/cyan] first.")
        raise typer.Exit(1)

    with cli_http_errors("session"):
        session = fetch_cogames_whoami(login_server=login_server, token=user_token)

    console.print("[green]Restored user session.[/green]")
    console.print(f"subject_type: {session.subject_type}")
    console.print(f"subject_id: {session.subject_id or '-'}")
    console.print(f"owner_user_id: {session.owner_user_id or '-'}")
