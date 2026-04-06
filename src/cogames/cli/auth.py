"""Compatibility wrappers for softmax auth commands."""

import importlib

import typer

from softmax.auth import DEFAULT_COGAMES_SERVER

auth_app = typer.Typer(
    help="Compatibility wrappers for softmax auth commands",
    context_settings={"help_option_names": ["-h", "--help"]},
    no_args_is_help=True,
    rich_markup_mode="rich",
)


def _softmax_cli():
    # Import lazily so the standalone cogames CLI does not form a cycle through
    # softmax -> cogames.softmax_cli -> cogames.main during startup.
    return importlib.import_module("softmax.cli")


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
    """Compatibility wrapper for softmax login."""
    _softmax_cli().login_cmd(login_server=login_server, no_browser=no_browser, force=force)


@auth_app.command(name="logout")
def logout_cmd(
    login_server: str = typer.Option(
        DEFAULT_COGAMES_SERVER,
        "--login-server",
        metavar="URL",
        help="Authentication server URL",
    ),
) -> None:
    """Compatibility wrapper for softmax logout."""
    _softmax_cli().logout_cmd(login_server=login_server)


@auth_app.command(name="get-login-url")
def get_login_url_cmd(
    login_server: str = typer.Option(
        DEFAULT_COGAMES_SERVER,
        "--login-server",
        metavar="URL",
        help="Authentication server URL",
    ),
) -> None:
    """Compatibility wrapper for softmax get-login-url."""
    _softmax_cli().get_login_url_cmd(login_server=login_server)


@auth_app.command(name="status")
def status_cmd(
    login_server: str = typer.Option(
        DEFAULT_COGAMES_SERVER,
        "--login-server",
        metavar="URL",
        help="Authentication server URL",
    ),
    server: str | None = typer.Option(
        None,
        "--server",
        "-s",
        metavar="URL",
        help="Tournament API server URL to check against (deprecated; ignored).",
    ),
) -> None:
    """Compatibility wrapper for softmax status."""
    _ = server
    _softmax_cli().status_cmd(login_server=login_server)


@auth_app.command(name="get-token")
def get_token_cmd(
    login_server: str = typer.Option(
        DEFAULT_COGAMES_SERVER,
        "--login-server",
        metavar="URL",
        help="Authentication server URL",
    ),
) -> None:
    """Compatibility wrapper for softmax get-token."""
    _softmax_cli().get_token_cmd(login_server=login_server)


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
    """Compatibility wrapper for softmax set-token."""
    _softmax_cli().set_token_cmd(token=token, login_server=login_server)
