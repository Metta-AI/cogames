"""cogames auth — token management commands."""

import typer

from cogames.cli.base import console
from cogames.cli.client import TournamentServerClient
from cogames.cli.login import DEFAULT_COGAMES_SERVER, CoGamesAuthenticator, perform_login
from cogames.cli.submit import DEFAULT_SUBMIT_SERVER

auth_app = typer.Typer(
    help="Manage authentication tokens",
    context_settings={"help_option_names": ["-h", "--help"]},
    no_args_is_help=True,
    rich_markup_mode="rich",
)


def _authenticator() -> CoGamesAuthenticator:
    return CoGamesAuthenticator()


@auth_app.command(name="login")
def login_cmd(
    server: str = typer.Option(
        DEFAULT_COGAMES_SERVER,
        "--login-server",
        metavar="URL",
        help="Authentication server URL",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Re-authenticate even if already logged in",
    ),
    timeout: int = typer.Option(
        300,
        "--timeout",
        "-t",
        metavar="SECS",
        help="Authentication timeout in seconds",
    ),
) -> None:
    """Authenticate with CoGames server via browser OAuth flow."""
    from urllib.parse import urlparse  # noqa: PLC0415

    auth = _authenticator()
    if auth.has_saved_token(server) and not force:
        console.print(f"[green]Already authenticated with {urlparse(server).hostname}[/green]")
        return

    console.print(f"[cyan]Authenticating with {server}...[/cyan]")
    if perform_login(auth_server_url=server, force=force, timeout=timeout):
        console.print("[green]Authentication successful![/green]")
    else:
        console.print("[red]Authentication failed![/red]")
        raise typer.Exit(1)


@auth_app.command(name="status")
def status_cmd(
    server: str = typer.Option(
        DEFAULT_COGAMES_SERVER,
        "--server",
        "-s",
        metavar="URL",
        help="Authentication server URL",
    ),
    api_server: str = typer.Option(
        DEFAULT_SUBMIT_SERVER,
        "--api-server",
        metavar="URL",
        help="API server URL to check against",
    ),
) -> None:
    """Check authentication status by calling /whoami."""
    client = TournamentServerClient.from_login(api_server, server)
    if not client:
        raise typer.Exit(1)

    result = client._get("/whoami")
    email = result.get("user_email", "unknown")
    console.print(f"[green]Authenticated as {email}[/green]")


@auth_app.command(name="get-token")
def get_token_cmd(
    server: str = typer.Option(
        DEFAULT_COGAMES_SERVER,
        "--server",
        "-s",
        metavar="URL",
        help="Authentication server URL",
    ),
) -> None:
    """Print the saved token to stdout (for scripting)."""
    auth = _authenticator()
    token = auth.load_token(server)
    if not token:
        console.print("[red]No token found.[/red] Run [cyan]cogames auth login[/cyan] first.", style="bold")
        raise typer.Exit(1)
    print(token)


@auth_app.command(name="set-token")
def set_token_cmd(
    token: str = typer.Argument(help="Bearer token to save"),
    server: str = typer.Option(
        DEFAULT_COGAMES_SERVER,
        "--server",
        "-s",
        metavar="URL",
        help="Authentication server URL",
    ),
) -> None:
    """Manually set a token (for CI or headless environments)."""
    auth = _authenticator()
    auth.config_reader_writer.save_token(token, server)
