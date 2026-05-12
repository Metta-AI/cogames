"""CoGames authentication commands."""

import importlib

import typer

from cogames.cli.player import player_app

auth_app = typer.Typer(
    help="CoGames authentication commands",
    context_settings={"help_option_names": ["-h", "--help"]},
    no_args_is_help=True,
    rich_markup_mode="rich",
)
auth_app.add_typer(player_app, name="player")


def _softmax_cli():
    # Import lazily so the standalone cogames CLI does not form a cycle through
    # softmax -> cogames.softmax_cli -> cogames.main during startup.
    return importlib.import_module("softmax.cli")


@auth_app.command(name="login")
def login_cmd(
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
    server: str | None = typer.Option(
        None,
        "--server",
        "-s",
        metavar="URL",
        help="Authentication server URL.",
    ),
) -> None:
    """Log in to CoGames."""
    _softmax_cli().login_cmd(no_browser=no_browser, force=force, server=server)


@auth_app.command(name="logout")
def logout_cmd(
    server: str | None = typer.Option(
        None,
        "--server",
        "-s",
        metavar="URL",
        help="Authentication server URL.",
    ),
) -> None:
    """Log out of CoGames."""
    _softmax_cli().logout_cmd(server=server)


@auth_app.command(name="get-login-url")
def get_login_url_cmd(
    server: str | None = typer.Option(
        None,
        "--server",
        "-s",
        metavar="URL",
        help="Authentication server URL.",
    ),
) -> None:
    """Print the CoGames login URL."""
    _softmax_cli().get_login_url_cmd(server=server)


@auth_app.command(name="status")
def status_cmd(
    server: str | None = typer.Option(
        None,
        "--server",
        "-s",
        metavar="URL",
        help="Authentication server URL.",
    ),
) -> None:
    """Show CoGames authentication status."""
    _softmax_cli().status_cmd(server=server)


@auth_app.command(name="get-token")
def get_token_cmd(
    server: str | None = typer.Option(
        None,
        "--server",
        "-s",
        metavar="URL",
        help="Authentication server URL.",
    ),
) -> None:
    """Print the saved CoGames token."""
    _softmax_cli().get_token_cmd(server=server)


@auth_app.command(name="set-token")
def set_token_cmd(
    token: str = typer.Argument(help="Bearer token to save"),
    server: str | None = typer.Option(
        None,
        "--server",
        "-s",
        metavar="URL",
        help="Authentication server URL.",
    ),
) -> None:
    """Save a CoGames token."""
    _softmax_cli().set_token_cmd(token=token, server=server)
