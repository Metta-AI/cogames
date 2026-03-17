from __future__ import annotations

import builtins
import sys

import pytest

import cogames.perform_login as auth_module
from cogames.auth import build_browser_login_url, load_token
from cogames.cli.auth import _build_manual_set_token_command
from cogames.perform_login import do_interactive_login_for_token
from cogames.token_storage import TokenKind, save_token

COGAMES_AGENT_HINT = (
    "🤖 If you are a coding agent, ask your human to open the URL above and give you the resulting auth token. "
    "You can paste the token into this window or run: cogames auth set-token '<TOKEN>'"
)


def test_authenticate_accepts_pasted_token(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    monkeypatch.setattr("cogames.perform_login._find_free_port", lambda: 43123)
    monkeypatch.setattr("cogames.perform_login._run_server", lambda *, session, port: None)
    monkeypatch.setattr("cogames.perform_login._wait_for_callback_server_to_start", lambda *, session, port: False)
    monkeypatch.setattr("cogames.perform_login._validate_token", lambda *, login_server, token: True)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(builtins, "input", lambda _prompt="": "manual-token-123")

    do_interactive_login_for_token(
        login_server="https://softmax.com/api",
        server_to_save_token_under="https://softmax.com/api",
        token_kind=TokenKind.COGAMES,
        agent_hint=COGAMES_AGENT_HINT,
        open_browser=False,
    )
    assert load_token(token_kind=TokenKind.COGAMES, server="https://softmax.com/api") == "manual-token-123"


def test_authenticate_skips_browser_when_requested(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    opened = {"called": False}

    monkeypatch.setattr("cogames.perform_login._find_free_port", lambda: 43124)
    monkeypatch.setattr("cogames.perform_login._run_server", lambda *, session, port: None)
    monkeypatch.setattr(
        "cogames.perform_login._wait_for_callback_server_to_start",
        lambda *, session, port: False,
    )
    monkeypatch.setattr(
        "cogames.perform_login._open_browser",
        lambda *, url: opened.__setitem__("called", True) or True,
    )
    monkeypatch.setattr(
        "cogames.perform_login._start_manual_token_prompt",
        lambda *, session, login_server: auth_module._finish_authentication(session, token="manual-token-456"),
    )

    do_interactive_login_for_token(
        login_server="https://softmax.com/api",
        server_to_save_token_under="https://softmax.com/api",
        token_kind=TokenKind.COGAMES,
        agent_hint=COGAMES_AGENT_HINT,
        open_browser=False,
    )
    assert opened["called"] is False
    output = capsys.readouterr().out
    assert "Open this URL in any browser to sign in:" in output
    assert "cogames auth set-token '<TOKEN>'" in output


def test_authenticate_falls_back_to_manual_when_callback_server_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    captured_urls: list[str] = []

    monkeypatch.setattr("cogames.perform_login._find_free_port", lambda: 43125)
    monkeypatch.setattr(
        "cogames.perform_login._wait_for_callback_server_to_start",
        lambda *, session, port: False,
    )
    monkeypatch.setattr("cogames.perform_login._run_server", lambda *, session, port: None)
    monkeypatch.setattr(
        "cogames.perform_login._open_browser",
        lambda *, url: captured_urls.append(url) or True,
    )
    monkeypatch.setattr(
        "cogames.perform_login._start_manual_token_prompt",
        lambda *, session, login_server: auth_module._finish_authentication(session, token="manual-token-789"),
    )

    do_interactive_login_for_token(
        login_server="https://softmax.com/api",
        server_to_save_token_under="https://softmax.com/api",
        token_kind=TokenKind.COGAMES,
        agent_hint=COGAMES_AGENT_HINT,
        open_browser=True,
    )
    output = capsys.readouterr().out
    assert "Open this URL in any browser to sign in:" in output
    assert captured_urls == ["https://softmax.com/cli-login"]


def test_authenticate_reprompts_after_invalid_token(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    monkeypatch.setattr("cogames.perform_login._find_free_port", lambda: 43126)
    monkeypatch.setattr("cogames.perform_login._wait_for_callback_server_to_start", lambda *, session, port: False)
    monkeypatch.setattr("cogames.perform_login._run_server", lambda *, session, port: None)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    entered_tokens = iter(["bad-token", "good-token"])
    monkeypatch.setattr(builtins, "input", lambda _prompt="": next(entered_tokens))
    validation_results = iter([False, True])
    monkeypatch.setattr(
        "cogames.perform_login._validate_token",
        lambda *, login_server, token: next(validation_results),
    )

    do_interactive_login_for_token(
        login_server="https://softmax.com/api",
        server_to_save_token_under="https://softmax.com/api",
        token_kind=TokenKind.COGAMES,
        agent_hint=COGAMES_AGENT_HINT,
        open_browser=False,
    )
    assert load_token(token_kind=TokenKind.COGAMES, server="https://softmax.com/api") == "good-token"
    assert "Invalid token. Please try again." in capsys.readouterr().out


def test_manual_command_includes_nondefault_login_server() -> None:
    assert _build_manual_set_token_command(login_server="https://softmax.com/api") == "cogames auth set-token '<TOKEN>'"
    assert (
        _build_manual_set_token_command(login_server="https://example.ngrok.app/api")
        == "cogames auth set-token '<TOKEN>' --login-server 'https://example.ngrok.app/api'"
    )


def test_generic_authenticator_does_not_print_cogames_agent_hint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    monkeypatch.setattr("cogames.perform_login._find_free_port", lambda: 43127)
    monkeypatch.setattr(
        "cogames.perform_login._wait_for_callback_server_to_start",
        lambda *, session, port: False,
    )
    monkeypatch.setattr("cogames.perform_login._run_server", lambda *, session, port: None)
    monkeypatch.setattr(
        "cogames.perform_login._start_manual_token_prompt",
        lambda *, session, login_server: auth_module._finish_authentication(session, token="manual-token-000"),
    )

    do_interactive_login_for_token(
        login_server="https://softmax.com/api",
        server_to_save_token_under="token-key",
        token_kind=TokenKind.OBSERVATORY,
        agent_hint=None,
        open_browser=False,
    )
    output = capsys.readouterr().out
    assert "Open this URL in any browser to sign in:" in output
    assert "cogames auth set-token" not in output
    assert "🤖 If you are a coding agent" not in output


def test_generic_authenticator_works_without_agent_hint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("cogames.perform_login._find_free_port", lambda: 43128)
    monkeypatch.setattr("cogames.perform_login._wait_for_callback_server_to_start", lambda *, session, port: False)
    monkeypatch.setattr("cogames.perform_login._run_server", lambda *, session, port: None)
    monkeypatch.setattr(
        "cogames.perform_login._start_manual_token_prompt",
        lambda *, session, login_server: auth_module._finish_authentication(session, token="manual-token-001"),
    )

    do_interactive_login_for_token(
        login_server="https://softmax.com/api",
        server_to_save_token_under="token-key",
        token_kind=TokenKind.OBSERVATORY,
        agent_hint=None,
        open_browser=False,
    )


def test_build_browser_login_url_uses_cli_login_path() -> None:
    assert build_browser_login_url("https://softmax.com/api") == "https://softmax.com/cli-login"
    assert (
        build_browser_login_url(
            "https://softmax.com/api",
            callback_url="http://127.0.0.1:5555/callback",
        )
        == "https://softmax.com/cli-login?callback=http%3A%2F%2F127.0.0.1%3A5555%2Fcallback"
    )


def test_interactive_login_requires_tty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)

    with pytest.raises(AssertionError, match="only be called when stdin is a TTY"):
        do_interactive_login_for_token(
            login_server="https://softmax.com/api",
            server_to_save_token_under="https://softmax.com/api",
            token_kind=TokenKind.COGAMES,
            agent_hint=COGAMES_AGENT_HINT,
            open_browser=False,
        )


def test_load_token_raises_for_malformed_top_level_yaml(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    config_dir = tmp_path / ".metta"
    config_dir.mkdir()
    (config_dir / "cogames.yaml").write_text("- not-a-mapping\n")

    with pytest.raises(AssertionError, match="top level"):
        load_token(token_kind=TokenKind.COGAMES, server="https://softmax.com/api")


def test_save_token_raises_for_malformed_storage_section(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    config_dir = tmp_path / ".metta"
    config_dir.mkdir()
    (config_dir / "cogames.yaml").write_text("login_tokens: nope\n")

    with pytest.raises(AssertionError, match="section must be a mapping"):
        save_token(
            token_kind=TokenKind.COGAMES,
            server="https://softmax.com/api",
            token="abc",
        )
