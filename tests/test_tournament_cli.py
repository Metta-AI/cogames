"""Regression tests for tournament CLI commands.

Covers:
- Top-level tournament command entrypoints are available.
- JSON output stability (valid JSON, expected fields).
- Auth-required vs public-read command behavior.
- Help text content for grouped commands.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import click
import pytest
from _test_support import (
    TEST_UUID,
    TEST_UUID_2,
    TEST_UUID_3,
    TEST_UUID_4,
    leaderboard_entry,
    match_response,
    policy_version_row,
    policy_version_summary,
    season_detail,
    season_summary,
)
from pytest_httpserver import HTTPServer
from typer.main import get_command
from typer.testing import CliRunner

from cogames.cli.client import TournamentServerClient
from cogames.cli.generated_models import MatchPlayerInfo
from cogames.cli.submit import observatory_profile_url
from cogames.main import _submit_browser_launch_skip_reason, app
from softmax.auth import save_token
from softmax.token_storage import TokenKind

_SEASON_ID = str(TEST_UUID)
_POLICY_VERSION_ID = str(TEST_UUID_2)
_POLICY_ID = str(TEST_UUID_3)
_MATCH_ID = str(TEST_UUID_4)

runner = CliRunner()


@pytest.fixture(autouse=True)
def _disable_display_for_cli_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("cogames.main.has_display", lambda: False)


def _season_summary(
    name: str = "test-season",
    is_default: bool = True,
) -> dict[str, Any]:
    return season_summary(
        name=name,
        display_name=name,
        is_default=is_default,
        created_at="2026-02-25T12:00:00Z",
        entry_pool=None,
        leaderboard_pool="ranked",
        pools=[{"name": "ranked", "description": "ranked pool", "config_id": None}],
    ).model_dump(mode="json")


def _season_info(name: str = "test-season") -> dict[str, Any]:
    return season_detail(
        name=name,
        display_name=name,
        created_at="2026-02-25T12:00:00Z",
        entry_pool=None,
        leaderboard_pool="ranked",
        pools=[{"name": "ranked", "description": "ranked pool", "config_id": None}],
        entrant_count=5,
        active_entrant_count=3,
        match_count=10,
        stage_count=1,
    ).model_dump(mode="json")


def _leaderboard_entries() -> list[dict[str, Any]]:
    return [
        leaderboard_entry(
            rank=1,
            policy=policy_version_summary(id=TEST_UUID_2, name="top-policy", version=3),
            score=42.5,
            score_stddev=1.2,
            matches=20,
        ).model_dump(mode="json"),
        leaderboard_entry(
            rank=2,
            policy=policy_version_summary(id=uuid.uuid4(), name="runner-up", version=1),
            score=38.0,
            score_stddev=2.1,
            matches=18,
        ).model_dump(mode="json"),
    ]


def _match_response() -> dict[str, Any]:
    return match_response(
        id=TEST_UUID_4,
        pool_name="ranked",
        created_at="2026-02-25T12:00:00+00:00",
        players=[
            MatchPlayerInfo(
                policy=policy_version_summary(id=TEST_UUID_2, name="my-policy", version=1),
                num_agents=2,
                score=10.0,
            ),
            MatchPlayerInfo(
                policy=policy_version_summary(id=uuid.uuid4(), name="opponent", version=1),
                num_agents=2,
                score=5.0,
            ),
        ],
    ).model_dump(mode="json")


def _policy_version_info() -> dict[str, Any]:
    return policy_version_row(
        id=TEST_UUID_2,
        policy_id=TEST_UUID_3,
        created_at="2026-02-25T12:00:00+00:00",
        policy_created_at="2026-02-25T11:00:00+00:00",
    ).model_dump(mode="json")


def _setup_read_endpoints(httpserver: HTTPServer) -> None:
    httpserver.expect_request(
        "/tournament/seasons",
        method="GET",
    ).respond_with_json([_season_summary()])
    httpserver.expect_request(
        "/tournament/seasons/test-season",
        method="GET",
    ).respond_with_json(_season_info())
    httpserver.expect_request(
        "/tournament/seasons/test-season/leaderboard",
        method="GET",
    ).respond_with_json(_leaderboard_entries())
    httpserver.expect_request(
        "/tournament/seasons/test-season/versions",
        method="GET",
    ).respond_with_json(
        [
            {"version": 1, "canonical": True, "disabled_at": None, "created_at": "2026-02-25T12:00:00Z"},
        ]
    )


def _mock_from_login(httpserver: HTTPServer):
    """Patch from_login to return a client pointing at the test httpserver."""

    def fake_from_login(server_url: str, login_server: str) -> TournamentServerClient:
        return TournamentServerClient(server_url=server_url, token="fake-token")

    return patch.object(TournamentServerClient, "from_login", side_effect=fake_from_login)


def _invoke_with_server(httpserver: HTTPServer, *args: str):
    return runner.invoke(
        app,
        [*args, "--server", httpserver.url_for(""), "--login-server", "http://fake-login-server"],
    )


# ---------------------------------------------------------------------------
# Top-level command regression tests
# ---------------------------------------------------------------------------


class TestTournamentCommands:
    """Verify top-level command names and routing."""

    def test_season_command_exists(self) -> None:
        result = runner.invoke(app, ["season", "--help"])
        assert result.exit_code == 0
        assert "season" in result.output.lower()

    def test_leaderboard_command_exists(self) -> None:
        result = runner.invoke(app, ["leaderboard", "--help"])
        assert result.exit_code == 0
        assert "leaderboard" in result.output.lower()

    def test_matches_command_exists(self) -> None:
        result = runner.invoke(app, ["matches", "--help"])
        assert result.exit_code == 0
        assert "match" in result.output.lower()

    def test_submissions_command_exists(self) -> None:
        result = runner.invoke(app, ["submissions", "--help"])
        assert result.exit_code == 0
        assert "submission" in result.output.lower() or "upload" in result.output.lower()

    def test_upload_command_exists(self) -> None:
        result = runner.invoke(app, ["upload", "--help"])
        assert result.exit_code == 0
        assert "upload" in result.output.lower()

    def test_submit_command_exists(self) -> None:
        result = runner.invoke(app, ["submit", "--help"])
        assert result.exit_code == 0
        assert "submit" in result.output.lower()

    def test_hidden_alias_games_exists(self) -> None:
        result = runner.invoke(app, ["games", "--help"])
        assert result.exit_code == 0

    def test_hidden_alias_mission_exists(self) -> None:
        result = runner.invoke(app, ["mission", "--help"])
        assert result.exit_code == 0

    def test_main_help_shows_tournament_commands(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        output = result.output
        assert "season" in output
        assert "leaderboard" in output
        assert "matches" in output
        assert "upload" in output
        assert "submit" in output

    def test_season_list_returns_data(self, httpserver: HTTPServer) -> None:
        _setup_read_endpoints(httpserver)
        result = _invoke_with_server(httpserver, "season", "list", "--json")
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data) == 1
        assert data[0]["name"] == "test-season"

    def test_season_show_returns_data(self, httpserver: HTTPServer) -> None:
        _setup_read_endpoints(httpserver)
        result = _invoke_with_server(httpserver, "season", "show", "test-season", "--json")
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, dict)
        assert data["name"] == "test-season"
        assert data["status"] == "in_progress"

    def test_season_versions_requires_name(self, httpserver: HTTPServer) -> None:
        _setup_read_endpoints(httpserver)
        result = _invoke_with_server(httpserver, "season", "versions")
        assert result.exit_code == 2
        assert "missing argument 'season'" in result.output.lower()

    def test_leaderboard_returns_data(self, httpserver: HTTPServer) -> None:
        _setup_read_endpoints(httpserver)
        result = _invoke_with_server(httpserver, "leaderboard", "--season", "test-season", "--json")
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data) == 2
        assert data[0]["rank"] == 1

    def test_leaderboard_uses_server_default_when_omitted(self, httpserver: HTTPServer) -> None:
        _setup_read_endpoints(httpserver)
        result = _invoke_with_server(httpserver, "leaderboard", "--json")
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data) == 2
        assert data[0]["rank"] == 1

    def test_matches_help_shows_examples(self) -> None:
        result = runner.invoke(app, ["matches", "--help"])
        assert result.exit_code == 0
        assert "cogames matches" in result.output

    def test_submissions_help_shows_examples(self) -> None:
        result = runner.invoke(app, ["submissions", "--help"])
        assert result.exit_code == 0
        assert "cogames submissions" in result.output

    def test_season_help_shows_commands(self) -> None:
        result = runner.invoke(app, ["season", "--help"])
        assert result.exit_code == 0
        assert "list" in result.output.lower()
        assert "show" in result.output.lower()

    def test_leaderboard_help_shows_examples(self) -> None:
        result = runner.invoke(app, ["leaderboard", "--help"])
        assert result.exit_code == 0
        assert "cogames leaderboard" in result.output


# ---------------------------------------------------------------------------
# JSON output stability tests
# ---------------------------------------------------------------------------


class TestJsonOutputStability:
    """Verify --json flag produces valid JSON with expected schema fields."""

    def test_season_list_json_output(self, httpserver: HTTPServer) -> None:
        _setup_read_endpoints(httpserver)
        result = _invoke_with_server(httpserver, "season", "list", "--json")
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)
        assert len(data) == 1
        season = data[0]
        assert "name" in season
        assert "version" in season
        assert "canonical" in season
        assert "is_default" in season
        assert season["name"] == "test-season"

    def test_season_list_json_has_pools(self, httpserver: HTTPServer) -> None:
        _setup_read_endpoints(httpserver)
        result = _invoke_with_server(httpserver, "season", "list", "--json")
        data = json.loads(result.output)
        season = data[0]
        assert "pools" in season
        assert isinstance(season["pools"], list)
        assert season["pools"][0]["name"] == "ranked"

    def test_season_versions_json_output(self, httpserver: HTTPServer) -> None:
        _setup_read_endpoints(httpserver)
        result = _invoke_with_server(httpserver, "season", "versions", "test-season", "--json")
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)
        assert len(data) == 1
        version = data[0]
        assert "version" in version
        assert "canonical" in version

    def test_leaderboard_json_output(self, httpserver: HTTPServer) -> None:
        _setup_read_endpoints(httpserver)
        result = _invoke_with_server(httpserver, "leaderboard", "--season", "test-season", "--json")
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)
        assert len(data) == 2
        entry = data[0]
        assert "rank" in entry
        assert "policy" in entry
        assert "score" in entry
        assert "matches" in entry
        assert entry["policy"]["name"] == "top-policy"
        assert entry["rank"] == 1

    def test_leaderboard_json_policy_fields(self, httpserver: HTTPServer) -> None:
        _setup_read_endpoints(httpserver)
        result = _invoke_with_server(httpserver, "leaderboard", "--season", "test-season", "--json")
        data = json.loads(result.output)
        policy = data[0]["policy"]
        assert "id" in policy
        assert "name" in policy
        assert "version" in policy

    def test_leaderboard_json_has_score_stddev(self, httpserver: HTTPServer) -> None:
        _setup_read_endpoints(httpserver)
        result = runner.invoke(
            app,
            [
                "leaderboard",
                "--season",
                "test-season",
                "--json",
                "--server",
                httpserver.url_for(""),
                "--login-server",
                "http://fake-login-server",
            ],
        )
        data = json.loads(result.output)
        assert "score_stddev" in data[0]
        assert data[0]["score_stddev"] == 1.2

    def test_matches_json_output(self, httpserver: HTTPServer) -> None:
        _setup_read_endpoints(httpserver)
        httpserver.expect_request(
            "/stats/policy-versions",
            method="GET",
        ).respond_with_json({"entries": [_policy_version_info()], "total_count": 1})
        httpserver.expect_request(
            "/tournament/seasons/test-season/matches",
            method="GET",
        ).respond_with_json([_match_response()])

        with _mock_from_login(httpserver):
            result = _invoke_with_server(httpserver, "matches", "--json")
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)
        assert len(data) == 1
        match = data[0]
        assert "id" in match
        assert "season_name" in match
        assert "pool_name" in match
        assert "status" in match
        assert "players" in match
        assert match["status"] == "completed"

    def test_matches_json_player_fields(self, httpserver: HTTPServer) -> None:
        _setup_read_endpoints(httpserver)
        httpserver.expect_request(
            "/stats/policy-versions",
            method="GET",
        ).respond_with_json({"entries": [_policy_version_info()], "total_count": 1})
        httpserver.expect_request(
            "/tournament/seasons/test-season/matches",
            method="GET",
        ).respond_with_json([_match_response()])

        with _mock_from_login(httpserver):
            result = runner.invoke(
                app,
                [
                    "matches",
                    "--json",
                    "--server",
                    httpserver.url_for(""),
                    "--login-server",
                    "http://fake-login-server",
                ],
            )
        data = json.loads(result.output)
        player = data[0]["players"][0]
        assert "policy" in player
        assert "num_agents" in player
        assert "score" in player

    def test_matches_json_has_created_at(self, httpserver: HTTPServer) -> None:
        _setup_read_endpoints(httpserver)
        httpserver.expect_request(
            "/stats/policy-versions",
            method="GET",
        ).respond_with_json({"entries": [_policy_version_info()], "total_count": 1})
        httpserver.expect_request(
            "/tournament/seasons/test-season/matches",
            method="GET",
        ).respond_with_json([_match_response()])

        with _mock_from_login(httpserver):
            result = runner.invoke(
                app,
                [
                    "matches",
                    "--json",
                    "--server",
                    httpserver.url_for(""),
                    "--login-server",
                    "http://fake-login-server",
                ],
            )
        data = json.loads(result.output)
        assert "created_at" in data[0]

    def test_match_detail_json_output(self, httpserver: HTTPServer) -> None:
        _setup_read_endpoints(httpserver)
        httpserver.expect_request(
            f"/tournament/matches/{_MATCH_ID}",
            method="GET",
        ).respond_with_json(_match_response())

        with _mock_from_login(httpserver):
            result = _invoke_with_server(httpserver, "matches", _MATCH_ID, "--json")
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, dict)
        assert data["id"] == _MATCH_ID
        assert data["status"] == "completed"

    def test_submissions_json_output(self, httpserver: HTTPServer) -> None:
        _setup_read_endpoints(httpserver)
        httpserver.expect_request(
            "/stats/policy-versions",
            method="GET",
        ).respond_with_json({"entries": [_policy_version_info()], "total_count": 1})
        httpserver.expect_request(
            "/tournament/my-memberships",
            method="GET",
        ).respond_with_json({_POLICY_VERSION_ID: ["test-season"]})

        with _mock_from_login(httpserver):
            result = _invoke_with_server(httpserver, "submissions", "--json")
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)
        assert len(data) == 1
        entry = data[0]
        assert "name" in entry
        assert "version" in entry
        assert entry["name"] == "my-policy"

    def test_submissions_season_json_output(self, httpserver: HTTPServer) -> None:
        _setup_read_endpoints(httpserver)
        httpserver.expect_request(
            "/tournament/seasons/test-season/policies",
            method="GET",
        ).respond_with_json(
            [
                {
                    "policy": {"id": _POLICY_VERSION_ID, "name": "my-policy", "version": 1},
                    "pools": [{"pool_name": "ranked", "active": True, "completed": 5, "failed": 0, "pending": 1}],
                    "entered_at": "2026-02-25T12:00:00Z",
                }
            ]
        )

        with _mock_from_login(httpserver):
            result = _invoke_with_server(httpserver, "submissions", "--season", "test-season", "--json")
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)
        assert len(data) == 1
        assert "policy" in data[0]
        assert "pools" in data[0]
        assert data[0]["policy"]["name"] == "my-policy"


# ---------------------------------------------------------------------------
# Auth behavior tests
# ---------------------------------------------------------------------------


class TestAuthBehavior:
    """Verify auth-required vs public-read command behavior."""

    def test_leaderboard_works_without_auth(self, httpserver: HTTPServer) -> None:
        _setup_read_endpoints(httpserver)
        result = _invoke_with_server(httpserver, "leaderboard", "--season", "test-season", "--json")
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data) == 2

    def test_season_list_works_without_auth(self, httpserver: HTTPServer) -> None:
        _setup_read_endpoints(httpserver)
        result = _invoke_with_server(httpserver, "season", "list", "--json")
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data) == 1

    def test_season_show_works_without_auth(self, httpserver: HTTPServer) -> None:
        _setup_read_endpoints(httpserver)
        result = _invoke_with_server(httpserver, "season", "show", "test-season", "--json")
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["name"] == "test-season"

    def test_season_versions_works_without_auth(self, httpserver: HTTPServer) -> None:
        _setup_read_endpoints(httpserver)
        result = _invoke_with_server(httpserver, "season", "versions", "test-season", "--json")
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data) == 1

    def test_season_show_no_auth_header_sent_for_public_read(self, httpserver: HTTPServer) -> None:
        _setup_read_endpoints(httpserver)
        _invoke_with_server(httpserver, "season", "show", "test-season", "--json")
        for req, _ in httpserver.log:
            if req.path == "/tournament/seasons/test-season":
                assert "X-Auth-Token" not in req.headers
                break

    def test_submissions_requires_auth(self, httpserver: HTTPServer) -> None:
        result = runner.invoke(
            app,
            [
                "submissions",
                "--server",
                httpserver.url_for(""),
                "--login-server",
                "http://nonexistent-login-server",
            ],
        )
        combined = result.output.lower()
        assert "not authenticated" in combined or "softmax login" in combined

    def test_matches_requires_auth(self, httpserver: HTTPServer) -> None:
        result = runner.invoke(
            app,
            [
                "matches",
                "--server",
                httpserver.url_for(""),
                "--login-server",
                "http://nonexistent-login-server",
            ],
        )
        combined = result.output.lower()
        assert "not authenticated" in combined or "softmax login" in combined

    def test_upload_requires_auth(self, httpserver: HTTPServer) -> None:
        _setup_read_endpoints(httpserver)
        result = runner.invoke(
            app,
            [
                "upload",
                "--policy",
                "class=cogames.policy.starter_agent.StarterPolicy",
                "--name",
                "test-policy",
                "--server",
                httpserver.url_for(""),
                "--login-server",
                "http://nonexistent-login-server",
                "--skip-validation",
            ],
        )
        combined = result.output.lower()
        assert "not authenticated" in combined or "softmax login" in combined

    def test_submit_requires_auth(self, httpserver: HTTPServer) -> None:
        _setup_read_endpoints(httpserver)
        result = runner.invoke(
            app,
            [
                "submit",
                "test-policy",
                "--season",
                "test-season",
                "--server",
                httpserver.url_for(""),
                "--login-server",
                "http://nonexistent-login-server",
            ],
        )
        combined = result.output.lower()
        assert "not authenticated" in combined or "softmax login" in combined

    def test_season_list_no_auth_header_sent_for_public_read(self, httpserver: HTTPServer) -> None:
        _setup_read_endpoints(httpserver)
        _invoke_with_server(httpserver, "season", "list", "--json")
        for req, _ in httpserver.log:
            if req.path == "/tournament/seasons":
                assert "X-Auth-Token" not in req.headers
                break

    def test_leaderboard_no_auth_header_sent_for_public_read(self, httpserver: HTTPServer) -> None:
        _setup_read_endpoints(httpserver)
        _invoke_with_server(httpserver, "leaderboard", "--season", "test-season", "--json")
        for req, _ in httpserver.log:
            if req.path == "/tournament/seasons/test-season/leaderboard":
                assert "X-Auth-Token" not in req.headers
                break

    def test_auth_required_error_message_is_actionable(self, httpserver: HTTPServer) -> None:
        result = runner.invoke(
            app,
            [
                "submissions",
                "--server",
                httpserver.url_for(""),
                "--login-server",
                "http://nonexistent-login-server",
            ],
        )
        assert "softmax login" in result.output.lower()

    def test_matches_sends_auth_header_when_authenticated(self, httpserver: HTTPServer) -> None:
        _setup_read_endpoints(httpserver)
        httpserver.expect_request(
            "/stats/policy-versions",
            method="GET",
        ).respond_with_json({"entries": [_policy_version_info()], "total_count": 1})
        httpserver.expect_request(
            "/tournament/seasons/test-season/matches",
            method="GET",
        ).respond_with_json([_match_response()])

        with _mock_from_login(httpserver):
            runner.invoke(
                app,
                [
                    "matches",
                    "--json",
                    "--server",
                    httpserver.url_for(""),
                    "--login-server",
                    "http://fake-login-server",
                ],
            )

        for req, _ in httpserver.log:
            if req.path == "/stats/policy-versions":
                assert req.headers.get("X-Auth-Token") == "fake-token"
                break


# ---------------------------------------------------------------------------
# Help text content tests
# ---------------------------------------------------------------------------


class TestHelpTextContent:
    """Verify help text is informative and consistent across commands."""

    @staticmethod
    def _invoke_help(*args: str) -> str:
        result = runner.invoke(
            app,
            [*args, "--help"],
            env={"COLUMNS": "120", "LINES": "40"},
            terminal_width=120,
        )
        assert result.exit_code == 0
        return result.output

    @staticmethod
    def _get_command(*names: str):
        command = cast(click.Group, get_command(app))
        for name in names:
            command = cast(click.Group, command.commands[name])
        return command

    @staticmethod
    def _option_names(*names: str) -> set[str]:
        command = TestHelpTextContent._get_command(*names)
        return {opt for param in command.params for opt in getattr(param, "opts", [])}

    def test_main_help_has_tournament_panel(self) -> None:
        result = runner.invoke(
            app,
            ["--help"],
            env={"COLUMNS": "120", "LINES": "40"},
            terminal_width=120,
        )
        assert result.exit_code == 0
        assert "Tournament" in result.output


# ---------------------------------------------------------------------------
# Season lookup auth header tests
# ---------------------------------------------------------------------------


def _save_token(tmp_path: Path, token: str, login_server: str) -> None:
    _ = tmp_path
    save_token(token_kind=TokenKind.COGAMES, token=token, server=login_server)


class TestSeasonLookupAuth:
    """Verify that upload and submit pass (or omit) the auth token when querying seasons.

    The _resolve_season helper now loads a saved token via the cogames token helpers and
    forwards it to TournamentServerClient so that private seasons (e.g. 'test-season'
    used by CI service accounts) are accessible. Public users with no saved token should
    still be able to resolve public seasons without any auth header.
    """

    def test_upload_sends_token_in_season_lookup_when_token_saved(
        self,
        httpserver: HTTPServer,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        _save_token(tmp_path, "service-token-xyz", "http://fake-login-server")
        _setup_read_endpoints(httpserver)
        monkeypatch.setattr("cogames.main.upload_policy", lambda **_: None)

        runner.invoke(
            app,
            [
                "upload",
                "--policy",
                "class=cogames.policy.starter_agent.StarterPolicy",
                "--name",
                "test-policy",
                "--server",
                httpserver.url_for(""),
                "--login-server",
                "http://fake-login-server",
                "--skip-validation",
            ],
        )

        season_reqs = [req for req, _ in httpserver.log if req.path == "/tournament/seasons"]
        assert season_reqs, "Expected a request to /tournament/seasons"
        assert season_reqs[0].headers.get("X-Auth-Token") == "service-token-xyz"

    def test_season_list_sends_no_token_when_saved(
        self,
        httpserver: HTTPServer,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        _save_token(tmp_path, "service-token-xyz", "http://fake-login-server")
        _setup_read_endpoints(httpserver)

        result = _invoke_with_server(httpserver, "season", "list", "--json")

        assert result.exit_code == 0
        season_reqs = [req for req, _ in httpserver.log if req.path == "/tournament/seasons"]
        assert season_reqs, "Expected a request to /tournament/seasons"
        assert "X-Auth-Token" not in season_reqs[0].headers

    def test_season_show_sends_no_token_when_saved(
        self,
        httpserver: HTTPServer,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        _save_token(tmp_path, "service-token-xyz", "http://fake-login-server")
        _setup_read_endpoints(httpserver)

        result = _invoke_with_server(httpserver, "season", "show", "test-season", "--json")

        assert result.exit_code == 0
        season_reqs = [req for req, _ in httpserver.log if req.path == "/tournament/seasons/test-season"]
        assert season_reqs, "Expected a request to /tournament/seasons/test-season"
        assert "X-Auth-Token" not in season_reqs[0].headers

    def test_upload_sends_no_token_in_season_lookup_when_absent(
        self,
        httpserver: HTTPServer,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        # No token saved — the cogames token lookup returns None
        _setup_read_endpoints(httpserver)
        monkeypatch.setattr("cogames.main.upload_policy", lambda **_: None)

        runner.invoke(
            app,
            [
                "upload",
                "--policy",
                "class=cogames.policy.starter_agent.StarterPolicy",
                "--name",
                "test-policy",
                "--server",
                httpserver.url_for(""),
                "--login-server",
                "http://fake-login-server",
                "--skip-validation",
            ],
        )

        season_reqs = [req for req, _ in httpserver.log if req.path == "/tournament/seasons"]
        assert season_reqs, "Expected a request to /tournament/seasons"
        assert "X-Auth-Token" not in season_reqs[0].headers

    def test_submit_sends_token_in_season_lookup_when_token_saved(
        self,
        httpserver: HTTPServer,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        _save_token(tmp_path, "service-token-xyz", "http://fake-login-server")
        _setup_read_endpoints(httpserver)

        policy_version_id = str(uuid.uuid4())
        policy_id = str(uuid.uuid4())
        httpserver.expect_request("/stats/policy-versions", method="GET").respond_with_json(
            {
                "entries": [
                    {
                        "id": policy_version_id,
                        "policy_id": policy_id,
                        "created_at": "2026-01-01T00:00:00Z",
                        "policy_created_at": "2026-01-01T00:00:00Z",
                        "user_id": "u1",
                        "name": "test-policy",
                        "version": 1,
                    }
                ],
                "total_count": 1,
            }
        )
        httpserver.expect_request("/tournament/seasons/test-season/submissions", method="POST").respond_with_json(
            {"pools": ["ranked"]}
        )

        with _mock_from_login(httpserver):
            result = runner.invoke(
                app,
                [
                    "submit",
                    "test-policy",
                    "--season",
                    "test-season",
                    "--server",
                    httpserver.url_for(""),
                    "--login-server",
                    "http://fake-login-server",
                ],
            )

        assert result.exit_code == 0
        season_reqs = [req for req, _ in httpserver.log if req.path == "/tournament/seasons/test-season"]
        assert season_reqs, "Expected a request to /tournament/seasons/test-season"
        assert season_reqs[0].headers.get("X-Auth-Token") == "service-token-xyz"

    def test_submit_sends_no_token_in_season_lookup_when_absent(
        self,
        httpserver: HTTPServer,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        # No token saved — season lookup runs unauthenticated, then from_login fails
        _setup_read_endpoints(httpserver)

        runner.invoke(
            app,
            [
                "submit",
                "test-policy",
                "--season",
                "test-season",
                "--server",
                httpserver.url_for(""),
                "--login-server",
                "http://fake-login-server",
            ],
        )

        season_reqs = [req for req, _ in httpserver.log if req.path == "/tournament/seasons/test-season"]
        assert season_reqs, "Expected a request to /tournament/seasons/test-season"
        assert "X-Auth-Token" not in season_reqs[0].headers


class TestSubmitProfileLaunch:
    def test_submit_browser_launch_allows_redirected_stdout_in_gui_session(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("cogames.main.has_display", lambda: True)
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("sys.stdout.isatty", lambda: False)

        assert _submit_browser_launch_skip_reason() is None

    def test_submit_browser_launch_reports_non_interactive_sessions(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("cogames.main.has_display", lambda: True)
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)

        assert _submit_browser_launch_skip_reason() == "non-interactive session detected"

    @pytest.mark.parametrize(
        ("login_server_url", "expected_base_url"),
        [
            (
                "https://softmax.com/api",
                "https://softmax.com",
            ),
            (
                "https://softmax.com/api/observatory",
                "https://softmax.com",
            ),
            (
                "http://localhost:3002",
                "http://localhost:3002",
            ),
        ],
    )
    def test_observatory_profile_url_uses_browser_origin_of_selected_login_server(
        self,
        login_server_url: str,
        expected_base_url: str,
    ) -> None:
        policy_version_id = uuid.uuid4()

        assert (
            observatory_profile_url(
                policy_version_id,
                login_server_url=login_server_url,
            )
            == f"{expected_base_url}/observatory/profile?policyVersionId={policy_version_id}"
        )
