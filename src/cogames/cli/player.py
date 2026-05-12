from __future__ import annotations

import typer
from rich.panel import Panel
from rich.table import Table

from cogames.cli.base import cli_http_errors, console, emit_json
from cogames.cli.client import TournamentServerClient
from softmax.auth import (
    DEFAULT_COGAMES_API_SERVER,
    fetch_cogames_whoami,
    get_login_server,
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


def _load_user_token_or_exit() -> str:
    user_token = load_cogames_user_token(login_server=get_login_server())
    if user_token is None:
        console.print("[red]No saved user session found.[/red] Run [cyan]cogames auth login[/cyan] first.")
        raise typer.Exit(1)
    return user_token


def _resolve_player(client: TournamentServerClient, player: str) -> str:
    """Resolve a player name or ID to a player ID."""
    if player.startswith("ply_"):
        return player
    players = client.list_players()
    matches = [p for p in players if p.name == player]
    if len(matches) == 1:
        return matches[0].id
    if len(matches) == 0:
        console.print(f"[red]No player found with name:[/red] {player}")
        raise typer.Exit(1)
    console.print(f"[red]Multiple players named:[/red] {player}")
    for m in matches:
        console.print(f"  {m.id}  ({m.created_at.isoformat()})")
    console.print("Use the player ID directly to disambiguate.")
    raise typer.Exit(1)


@player_app.command(name="list")
def list_players_cmd(
    server: str = typer.Option(
        DEFAULT_COGAMES_API_SERVER,
        "--server",
        "-s",
        metavar="URL",
        help="Tournament API server URL.",
    ),
    json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """List the players owned by the saved user session."""
    user_token = _load_user_token_or_exit()
    with cli_http_errors("players"):
        with TournamentServerClient(server_url=server, token=user_token) as client:
            players = client.list_players()

    if json:
        emit_json([player.model_dump(mode="json") for player in players])
        return

    if not players:
        console.print("[yellow]No players found.[/yellow]")
        return

    table = Table(show_header=True)
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Name")
    table.add_column("Created")
    for player in players:
        table.add_row(player.id, player.name, player.created_at.isoformat())
    console.print(table)


@player_app.command(name="login")
def login_player_cmd(
    player: str = typer.Argument(..., metavar="PLAYER", help="Player name or ID (ply_...)."),
    server: str = typer.Option(
        DEFAULT_COGAMES_API_SERVER,
        "--server",
        "-s",
        metavar="URL",
        help="Tournament API server URL.",
    ),
    json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Mint a short-lived player session token and make it the active CoGames session."""
    user_token = _load_user_token_or_exit()
    with cli_http_errors("player"):
        with TournamentServerClient(server_url=server, token=user_token) as client:
            player_id = _resolve_player(client, player)
            response = client.login_player(player_id)

    save_cogames_active_token(login_server=get_login_server(), token=response.token)

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
    server: str | None = typer.Option(
        None,
        "--server",
        "-s",
        metavar="URL",
        help="API server URL for /whoami verification.",
    ),
) -> None:
    """Restore the saved user session as the active CoGames session."""
    user_token = restore_cogames_user_session(login_server=get_login_server())
    if user_token is None:
        console.print("[red]No saved user session found.[/red] Run [cyan]cogames auth login[/cyan] first.")
        raise typer.Exit(1)

    with cli_http_errors("session"):
        session = fetch_cogames_whoami(api_server=server, token=user_token)

    console.print("[green]Restored user session.[/green]")
    console.print(f"subject_type: {session.subject_type}")
    console.print(f"subject_id: {session.subject_id or '-'}")
    console.print(f"owner_user_id: {session.owner_user_id or '-'}")


@player_app.command(name="create")
def create_player_cmd(
    name: str = typer.Option(..., "--name", "-n", help="Name for the new player."),
    server: str = typer.Option(
        DEFAULT_COGAMES_API_SERVER,
        "--server",
        "-s",
        metavar="URL",
        help="Tournament API server URL.",
    ),
    json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Create a new player."""
    user_token = _load_user_token_or_exit()
    with cli_http_errors("player"):
        with TournamentServerClient(server_url=server, token=user_token) as client:
            player = client.create_player(name)

    if json:
        emit_json(player.model_dump(mode="json"))
        return

    console.print(f"[green]Player created:[/green] {player.id}")
    console.print(f"name: {player.name}")


# -- credentials sub-commands ------------------------------------------------

credentials_app = typer.Typer(
    help="Manage player API credentials.",
    context_settings={"help_option_names": ["-h", "--help"]},
    no_args_is_help=True,
    rich_markup_mode="rich",
)
player_app.add_typer(credentials_app, name="credentials")


@credentials_app.command(name="list")
def list_credentials_cmd(
    player: str = typer.Argument(..., metavar="PLAYER", help="Player name or ID (ply_...)."),
    server: str = typer.Option(
        DEFAULT_COGAMES_API_SERVER,
        "--server",
        "-s",
        metavar="URL",
        help="Tournament API server URL.",
    ),
    json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """List credentials for a player."""
    user_token = _load_user_token_or_exit()
    with cli_http_errors("credentials"):
        with TournamentServerClient(server_url=server, token=user_token) as client:
            player_id = _resolve_player(client, player)
            credentials = client.list_player_credentials(player_id)

    if json:
        emit_json([c.model_dump(mode="json") for c in credentials])
        return

    if not credentials:
        console.print("[yellow]No credentials found.[/yellow]")
        return

    table = Table(show_header=True)
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Name")
    table.add_column("Preview")
    table.add_column("Scopes")
    table.add_column("Created")
    for cred in credentials:
        table.add_row(
            str(cred.id),
            cred.name,
            cred.token_preview,
            ", ".join(cred.scopes) or "-",
            cred.created_at.isoformat(),
        )
    console.print(table)


@credentials_app.command(name="create")
def create_credential_cmd(
    player: str = typer.Argument(..., metavar="PLAYER", help="Player name or ID (ply_...)."),
    name: str = typer.Option(..., "--name", "-n", help="Name for the credential."),
    scope: list[str] = typer.Option([], "--scope", help="Credential scope (can be repeated). Valid: write"),
    server: str = typer.Option(
        DEFAULT_COGAMES_API_SERVER,
        "--server",
        "-s",
        metavar="URL",
        help="Tournament API server URL.",
    ),
    json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Create an API credential for a player. The token is shown only once."""
    user_token = _load_user_token_or_exit()
    with cli_http_errors("credential"):
        with TournamentServerClient(server_url=server, token=user_token) as client:
            player_id = _resolve_player(client, player)
            cred = client.create_player_credential(player_id, name, scopes=scope or None)

    if json:
        emit_json(cred.model_dump(mode="json"))
        return

    console.print(f"[green]Credential created:[/green] {cred.name}")
    console.print(Panel(cred.token, title="Token (save this — it will not be shown again)", border_style="yellow"))
    console.print(f"id: {cred.id}")
    console.print(f"scopes: {', '.join(cred.scopes) or '-'}")


@credentials_app.command(name="revoke")
def revoke_credential_cmd(
    player: str = typer.Argument(..., metavar="PLAYER", help="Player name or ID (ply_...)."),
    credential_id: str = typer.Argument(..., metavar="CREDENTIAL_ID", help="Credential UUID to revoke."),
    server: str = typer.Option(
        DEFAULT_COGAMES_API_SERVER,
        "--server",
        "-s",
        metavar="URL",
        help="Tournament API server URL.",
    ),
) -> None:
    """Revoke a player credential."""
    user_token = _load_user_token_or_exit()
    with cli_http_errors("credential"):
        with TournamentServerClient(server_url=server, token=user_token) as client:
            player_id = _resolve_player(client, player)
            client.revoke_player_credential(player_id, credential_id)

    console.print(f"[green]Credential revoked:[/green] {credential_id}")
