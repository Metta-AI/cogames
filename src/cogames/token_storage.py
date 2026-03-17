import os
from enum import StrEnum
from pathlib import Path

import yaml


class TokenKind(StrEnum):
    COGAMES = "cogames"
    OBSERVATORY = "observatory"


def _token_file_name(*, token_kind: TokenKind) -> str:
    if token_kind == TokenKind.COGAMES:
        return "cogames.yaml"
    if token_kind == TokenKind.OBSERVATORY:
        return "config.yaml"
    raise AssertionError(f"Unhandled token kind: {token_kind}")


def _token_storage_key(*, token_kind: TokenKind) -> str | None:
    if token_kind == TokenKind.COGAMES:
        return "login_tokens"
    if token_kind == TokenKind.OBSERVATORY:
        return "observatory_tokens"
    raise AssertionError(f"Unhandled token kind: {token_kind}")


def _token_file_path(*, token_file_name: str) -> Path:
    config_dir = Path.home() / ".metta"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir / token_file_name


def _load_token_data(*, token_file_name: str) -> dict:
    token_file = _token_file_path(token_file_name=token_file_name)
    if not token_file.exists():
        return {}

    with open(token_file, "r") as f:
        return yaml.safe_load(f) or {}


def load_token(*, token_kind: TokenKind, server: str) -> str | None:
    data = _load_token_data(token_file_name=_token_file_name(token_kind=token_kind))
    assert isinstance(data, dict), "Token storage file must contain a mapping at the top level"

    token_storage_key = _token_storage_key(token_kind=token_kind)
    tokens = data.get(token_storage_key, {}) if token_storage_key else data
    assert isinstance(tokens, dict), "Token storage section must be a mapping"

    token = tokens.get(server)
    return token if isinstance(token, str) else None


def save_token(*, token_kind: TokenKind, server: str, token: str) -> None:
    token_file_name = _token_file_name(token_kind=token_kind)
    token_storage_key = _token_storage_key(token_kind=token_kind)
    data = _load_token_data(token_file_name=token_file_name)
    assert isinstance(data, dict), "Token storage file must contain a mapping at the top level"

    if token_storage_key:
        tokens = data.setdefault(token_storage_key, {})
        assert isinstance(tokens, dict), "Token storage section must be a mapping"
        tokens[server] = token
    else:
        data[server] = token

    token_file = _token_file_path(token_file_name=token_file_name)
    with open(token_file, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False)
    os.chmod(token_file, 0o600)
