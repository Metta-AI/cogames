"""Token storage and browser URL helpers for CLI auth."""

from __future__ import annotations

from urllib.parse import urlencode, urlsplit, urlunsplit

from cogames.token_storage import TokenKind
from cogames.token_storage import load_token as load_saved_token
from cogames.token_storage import save_token as save_stored_token

DEFAULT_COGAMES_SERVER = "https://softmax.com/api"


def build_browser_login_url(login_server: str, *, callback_url: str | None = None) -> str:
    """Build the hosted browser sign-in URL for CLI login."""
    params: dict[str, str] = {}
    if callback_url:
        params["callback"] = callback_url

    query = urlencode(params)
    parsed = urlsplit(login_server)
    browser_path = parsed.path.rstrip("/").removesuffix("/api") + "/cli-login"
    return urlunsplit((parsed.scheme, parsed.netloc, browser_path, query, ""))


def load_token(*, token_kind: TokenKind, server: str) -> str | None:
    return load_saved_token(token_kind=token_kind, server=server)


def has_saved_token(*, token_kind: TokenKind, server: str) -> bool:
    return load_token(token_kind=token_kind, server=server) is not None


def save_token(*, token_kind: TokenKind, server: str, token: str) -> None:
    save_stored_token(token_kind=token_kind, server=server, token=token)
