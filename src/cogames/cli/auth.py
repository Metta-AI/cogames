"""cogames auth — token management commands."""

import sys

import typer
from rich.panel import Panel

from cogames.auth import (
    DEFAULT_COGAMES_SERVER,
    build_browser_login_url,
    delete_token,
    has_saved_token,
    load_token,
    save_token,
)
from cogames.cli.base import console
from cogames.cli.client import TournamentServerClient
from cogames.cli.submit import DEFAULT_SUBMIT_SERVER
from cogames.perform_login import do_interactive_login_for_token
from cogames.token_storage import TokenKind

auth_app = typer.Typer(
    help="Manage authentication tokens",
    context_settings={"help_option_names": ["-h", "--help"]},
    no_args_is_help=True,
    rich_markup_mode="rich",
)


def _build_manual_set_token_command(*, login_server: str) -> str:
    command = "cogames auth set-token '<TOKEN>'"
    if login_server != DEFAULT_COGAMES_SERVER:
        command += f" --login-server '{login_server}'"
    return command


def _print_non_tty_login_instructions(*, login_server: str) -> None:
    auth_url = build_browser_login_url(login_server)
    console.print("Interactive login requires a TTY.", style="red")
    console.print()
    console.print("Open this URL in any browser to sign in:", style="yellow")
    console.print()
    console.print("    ", auth_url)
    console.print()
    console.print("Copy the auth token from the browser, then run:", style="yellow")
    console.print()
    console.print("    ", _build_manual_set_token_command(login_server=login_server))
    console.print()
    console.print(
        Panel(
            "If you are a coding agent, ask your human to open the URL above and give you the resulting auth token. "
            "Then run the set-token command above.",
            title="🤖 Agent Hint",
            border_style="cyan",
        )
    )


@auth_app.command(name="login")
def login_cmd(
    login_server: str = typer.Option(
        DEFAULT_COGAMES_SERVER,
        "--login-server",
        metavar="URL",
        help="Authentication server URL",
    ),
    no_browser: bool = typer.Option(
        False,
        "--no-browser",
        help="Skip opening browser automatically.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Re-authenticate even if already logged in",
    ),
) -> None:
    """Sign in to cogames interactively."""
    from urllib.parse import urlparse  # noqa: PLC0415

    if has_saved_token(token_kind=TokenKind.COGAMES, server=login_server) and not force:
        console.print(f"Already authenticated with {urlparse(login_server).hostname}", style="green")
        return

    if not sys.stdin.isatty():
        _print_non_tty_login_instructions(login_server=login_server)
        raise typer.Exit(1)

    try:
        do_interactive_login_for_token(
            login_server=login_server,
            server_to_save_token_under=login_server,
            token_kind=TokenKind.COGAMES,
            agent_hint=(
                "If you are a coding agent, ask your human to open the URL below and give you "
                "the auth token. Then paste the token into this window or run:\n"
                "\n"
                f"{_build_manual_set_token_command(login_server=login_server)}"
            ),
            open_browser=not no_browser,
        )
    except Exception as e:
        console.print(f"Error: {e}")
        console.print()
        console.print("Authentication failed.", style="red")
        raise typer.Exit(1) from e

    console.print("Authentication successful.", style="green")


@auth_app.command(name="logout")
def logout_cmd(
    login_server: str = typer.Option(
        DEFAULT_COGAMES_SERVER,
        "--login-server",
        metavar="URL",
        help="Authentication server URL",
    ),
) -> None:
    """Remove saved authentication token."""
    if delete_token(token_kind=TokenKind.COGAMES, server=login_server):
        console.print("Logged out.", style="green")
    else:
        console.print("No token found — already logged out.", style="yellow")


@auth_app.command(name="get-login-url")
def get_login_url_cmd(
    login_server: str = typer.Option(
        DEFAULT_COGAMES_SERVER,
        "--login-server",
        metavar="URL",
        help="Authentication server URL",
    ),
) -> None:
    """Print a browser sign-in URL for manual login."""
    print(build_browser_login_url(login_server))


@auth_app.command(name="status")
def status_cmd(
    login_server: str = typer.Option(
        DEFAULT_COGAMES_SERVER,
        "--login-server",
        metavar="URL",
        help="Authentication server URL",
    ),
    server: str = typer.Option(
        DEFAULT_SUBMIT_SERVER,
        "--server",
        "-s",
        metavar="URL",
        help="Tournament API server URL to check against",
    ),
) -> None:
    """Check authentication status by calling /whoami."""
    client = TournamentServerClient.from_login(server, login_server)
    if not client:
        raise typer.Exit(1)

    result = client._get("/whoami")
    email = result.get("user_email", "unknown")
    console.print(f"[green]Authenticated as {email}[/green]")


@auth_app.command(name="get-token")
def get_token_cmd(
    login_server: str = typer.Option(
        DEFAULT_COGAMES_SERVER,
        "--login-server",
        metavar="URL",
        help="Authentication server URL",
    ),
) -> None:
    """Print the saved token to stdout (for scripting)."""
    token = load_token(token_kind=TokenKind.COGAMES, server=login_server)
    if not token:
        console.print("[red]No token found.[/red] Run [cyan]cogames auth login[/cyan] first.", style="bold")
        raise typer.Exit(1)
    print(token)


@auth_app.command(name="set-token")
def set_token_cmd(
    token: str = typer.Argument(help="Bearer token to save"),
    login_server: str = typer.Option(
        DEFAULT_COGAMES_SERVER,
        "--login-server",
        metavar="URL",
        help="Authentication server URL",
    ),
) -> None:
    """Manually set a token (for CI or headless environments)."""
    save_token(token_kind=TokenKind.COGAMES, token=token, server=login_server)
    print(f"\nToken saved for {login_server}")
