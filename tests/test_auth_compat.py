from __future__ import annotations

import pytest
from typer.testing import CliRunner

from cogames.main import app
from softmax.auth import load_token
from softmax.token_storage import TokenKind

runner = CliRunner()


def test_legacy_login_alias_removed() -> None:
    command_names = [cmd.name for cmd in app.registered_commands]
    assert "login" not in command_names


def test_auth_login_help_exists() -> None:
    result = runner.invoke(app, ["auth", "login", "--help"])
    assert result.exit_code == 0
    assert "login" in result.output.lower()


def test_auth_subcommands_remain_available() -> None:
    result = runner.invoke(app, ["auth", "--help"])
    assert result.exit_code == 0
    assert "set-token" in result.output
    assert "status" in result.output


def test_auth_set_token_calls_through_to_softmax(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    result = runner.invoke(app, ["auth", "set-token", "compat-token-123"])

    assert result.exit_code == 0
    assert load_token(token_kind=TokenKind.COGAMES, server="https://softmax.com/api") == "compat-token-123"
